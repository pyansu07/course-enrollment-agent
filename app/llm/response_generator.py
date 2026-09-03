"""LLM-based response generators - replace string formatting in agent nodes."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.llm.prompts import ENROLLMENT_DRAFT_PROMPT, QA_ANSWER_PROMPT
from app.llm.types import EnrollmentDraftResult


class QaResponseGenerator:
    """Generates course Q&A answers using LLM with search results as context.

    This is the RAG generation step - the LLM receives course data as context
    and composes a natural language answer.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    async def generate(self, user_message: str, courses: list[dict]) -> str:
        if not courses:
            return "I couldn't find any courses matching your query."

        context = "\n".join(
            f"- {c['name']} ({c['level']}, {c['duration']}, taught by {c['instructor']}): "
            f"${c['price']:.2f} | seats available: {c['seats_available']} | {c['description']}"
            for c in courses
        )
        prompt = QA_ANSWER_PROMPT.format(course_context=context)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=user_message),
        ]
        response = await self._llm.ainvoke(messages)
        return str(response.content)


class EnrollmentDraftGenerator:
    """Extracts course and seat count from the user's enrollment request using LLM.

    Uses OpenAI's native structured output for guaranteed JSON schema compliance.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm.with_structured_output(EnrollmentDraftResult)

    async def generate(
        self, user_message: str, course_catalog: list[dict], history: list[dict] | None = None
    ) -> EnrollmentDraftResult:
        catalog_text = "\n".join(
            f"- id: {c['id']} | {c['name']} | {c['level']} | {c['duration']} | "
            f"${c['price']:.2f} | seats available: {c['seats_available']}"
            for c in course_catalog
        )
        if history:
            conv_text = "Conversation history:\n" + "\n".join(
                f"[{m['role']}] {m['content'][:300]}" for m in history[-6:]
            )
        else:
            conv_text = "(no previous conversation)"
        prompt = ENROLLMENT_DRAFT_PROMPT.format(
            conversation_history=conv_text, course_catalog=catalog_text
        )
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=user_message),
        ]
        return await self._llm.ainvoke(messages)
