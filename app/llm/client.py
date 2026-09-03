from langchain_openai import ChatOpenAI


class LlmClient:
    """Chat LLM client — wraps ChatOpenAI creation.

    Talks to any OpenAI-compatible chat endpoint. `base_url` selects the provider:
    left as None it hits OpenAI, pointed at Groq's /openai/v1 it hits Groq. The
    wire format is identical, so nothing downstream changes.

    Created once in di.py and injected into LLM components.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-120b",
        base_url: str | None = None,
    ) -> None:
        kwargs: dict = {
            "model": model,
            "api_key": api_key,
            "temperature": 0.3,
        }
        # Only pass base_url when set — ChatOpenAI treats an explicit None as
        # "no override" but an empty string would break URL construction.
        if base_url:
            kwargs["base_url"] = base_url

        self.chat_openai = ChatOpenAI(**kwargs)
