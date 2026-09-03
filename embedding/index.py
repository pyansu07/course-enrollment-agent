"""Offline embedding indexer — builds the ChromaDB course collection.

Run once (and again whenever the catalog changes) with `make index`. It reads the
canonical catalog, embeds each course with a local sentence-transformers model, and
upserts the vectors into a remote ChromaDB collection. The running app never builds
the index itself — it only queries the collection at request time.

Re-run this after changing EMBEDDING_MODEL: the collection is recreated from scratch,
which is required because Chroma fixes vector width at collection-creation time.

This separation mirrors a real RAG pipeline: indexing is a batch job, retrieval is
online. Keeping it out of the request path means startup stays fast and embeddings
are computed exactly once.

Usage:
    python -m embedding.index
"""

import os

import chromadb
from dotenv import load_dotenv

from app.course.catalog import load_courses
from app.course.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    MINILM_DIMENSIONS,
    LocalHuggingFaceEmbeddingFunction,
)


def course_to_text(course: dict) -> str:
    """Flatten a course into the text that gets embedded.

    Only the semantically meaningful fields are included — name, category, level,
    instructor and the free-text description. Price and seat count are noise for
    similarity search.
    """
    fields = ("name", "category", "level", "instructor", "description")
    return " | ".join(str(course.get(field, "")) for field in fields)


def build_index() -> None:
    # Inside Docker these come from docker-compose. Run from the host, they come from
    # .env — where CHROMA_PORT must be 8001, the port compose maps Chroma to.
    load_dotenv()

    host = os.environ.get("CHROMA_HOST", "localhost")
    port = int(os.environ.get("CHROMA_PORT", "8000"))
    collection_name = os.environ.get("CHROMA_COLLECTION", "courses")
    embedding_model = os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

    courses = load_courses()
    print(f"Loaded {len(courses)} courses from the catalog.")

    # Embeddings are computed locally by sentence-transformers — no API key needed.
    # First run downloads the model (~90MB) into the HuggingFace cache.
    print(f"Loading local embedding model '{embedding_model}' (first run downloads it)...")
    embedding_fn = LocalHuggingFaceEmbeddingFunction(model_name=embedding_model)

    # Fail loudly here if the model's width is not what the rest of the pipeline
    # assumes — a dimension mismatch otherwise surfaces as an opaque Chroma error.
    probe_dims = len(embedding_fn(["dimension probe"])[0])
    print(f"Embedding dimensions: {probe_dims}")
    if probe_dims != MINILM_DIMENSIONS:
        print(
            f"  note: expected {MINILM_DIMENSIONS} for all-MiniLM-L6-v2. "
            f"The collection will be built at {probe_dims} dims; the app must use "
            f"this same model or retrieval will fail."
        )

    client = chromadb.HttpClient(host=host, port=port)
    print(f"Connected to ChromaDB at {host}:{port}.")

    # Recreate the collection from scratch so re-indexing is idempotent. This is also
    # what makes an embedding-model change safe: Chroma fixes a collection's vector
    # width at creation, so a 1536-dim collection cannot be reused for 384-dim vectors.
    try:
        client.delete_collection(collection_name)
        print(f"Dropped existing collection '{collection_name}'.")
    except Exception:
        # Collection did not exist yet — nothing to drop on a first run.
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        # Cosine distance is the standard metric for normalized text embeddings.
        configuration={"hnsw": {"space": "cosine"}},
    )

    # Embedding happens here: passing documents lets the embedding function turn
    # each course text into a vector before storing it in the collection.
    collection.add(
        ids=[c["id"] for c in courses],
        documents=[course_to_text(c) for c in courses],
        metadatas=[{"name": c["name"], "category": c["category"], "level": c["level"]} for c in courses],
    )

    print(f"Indexed {collection.count()} courses into '{collection_name}' (model: {embedding_model}).")


if __name__ == "__main__":
    build_index()
