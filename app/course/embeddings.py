"""Local embedding function for the course vector store.

Wraps langchain's HuggingFaceEmbeddings (which runs sentence-transformers locally)
behind ChromaDB's EmbeddingFunction interface, so the indexer and the repository can
keep passing an `embedding_function` to Chroma exactly as before.

Why local: embeddings are the only part of the RAG pipeline that ran per-request
against a paid API. A 384-dimension MiniLM model runs on CPU in milliseconds and
costs nothing, which is a sensible trade for a catalog of this size.

IMPORTANT — dimensionality: this model produces 384-dim vectors. OpenAI's
text-embedding-3-small produced 1536. ChromaDB fixes a collection's dimensionality
at creation time, so switching models REQUIRES dropping and rebuilding the
collection (`make index` does this). Querying a 1536-dim collection with a 384-dim
vector fails; mixing them silently is worse, so the indexer always recreates.

The model is downloaded once (~90MB) to the HuggingFace cache, then loaded from disk.
"""

from functools import lru_cache

# Default matches .env.example. 384 dims, fast on CPU, good enough for short
# catalog descriptions.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# all-MiniLM-L6-v2 output width. Asserted after load so a model swap that changes
# dimensionality fails loudly here rather than as a confusing Chroma error later.
MINILM_DIMENSIONS = 384


@lru_cache(maxsize=2)
def _load_langchain_embeddings(model_name: str):
    """Load the sentence-transformers model via langchain (cached per model name).

    Imported lazily so that importing this module — and therefore the DI container —
    never pulls in torch or touches the HuggingFace cache.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=model_name,
        # Normalized vectors make cosine distance well behaved, which is the
        # metric the collection is configured with.
        encode_kwargs={"normalize_embeddings": True},
    )


class LocalHuggingFaceEmbeddingFunction:
    """ChromaDB-compatible embedding function backed by langchain HuggingFaceEmbeddings.

    Chroma calls this with a list of documents and expects a list of vectors. It also
    asks for `name()` so it can record which embedder built a collection.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self._model_name = model_name
        self._embeddings = None

    @staticmethod
    def name() -> str:
        """Identifier Chroma stores in the collection's embedding-function config."""
        return "local_huggingface"

    def _get_embeddings(self):
        if self._embeddings is None:
            self._embeddings = _load_langchain_embeddings(self._model_name)
        return self._embeddings

    def __call__(self, input: list[str]) -> list[list[float]]:
        # Chroma's interface names this parameter `input`; keep it for compatibility.
        return self._get_embeddings().embed_documents(list(input))

    def embed_query(self, input: list[str]):
        """Embed query text. Chroma calls this on the retrieval path.

        Note the shape: Chroma passes a LIST of query strings and expects vectors
        back — not a single string, despite the singular name. langchain's own
        embed_query is per-string, so map over the list. Chroma's fastapi client
        then calls `.tolist()` on each embedding, so these must be numpy arrays,
        not plain Python lists.
        """
        import numpy as np

        embeddings = self._get_embeddings()
        return [np.array(embeddings.embed_query(text)) for text in list(input)]

    # --- Chroma EF config plumbing -------------------------------------------------
    # Chroma serializes the embedding function alongside the collection so a later
    # get_collection can rebuild it. These three keep that round-trip working.

    def get_config(self) -> dict:
        return {"model_name": self._model_name}

    @staticmethod
    def build_from_config(config: dict) -> "LocalHuggingFaceEmbeddingFunction":
        return LocalHuggingFaceEmbeddingFunction(
            model_name=config.get("model_name", DEFAULT_EMBEDDING_MODEL)
        )

    def is_legacy(self) -> bool:
        return False
