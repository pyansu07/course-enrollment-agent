import os

# Must be set BEFORE importing app — LlmClient reads OPENAI_API_KEY at import time.
os.environ["OPENAI_API_KEY"] = "test-key"

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture(autouse=True)
def setup_graph_and_state() -> None:
    """Set up InMemorySaver checkpointer and clear enrollment repo between tests."""
    from langgraph.checkpoint.memory import InMemorySaver

    import app.main as main_module
    from app.config.di import agent_graph_builder, enrollment_repo

    main_module.agent = agent_graph_builder.build(InMemorySaver())
    enrollment_repo.reset()


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _mock_guard(input_on_topic: bool, output_valid: bool = True):
    """Mock both input and output guardrail checks."""
    from app.llm.types import InputGuardResult, OutputGuardResult

    async def check_input(self, message: str, history=None) -> InputGuardResult:
        reason = "ok" if input_on_topic else "off-topic"
        return InputGuardResult(on_topic=input_on_topic, reason=reason)

    async def check_output(self, answer: str) -> OutputGuardResult:
        return OutputGuardResult(valid=output_valid, reason="ok" if output_valid else "invalid")

    return (
        patch("app.llm.guardrail.Guardrail.check_input", check_input),
        patch("app.llm.guardrail.Guardrail.check_output", check_output),
    )


def _mock_skill(skill: str):
    from app.llm.types import SkillResult

    return patch(
        "app.llm.skill_router.SkillRouter.classify",
        AsyncMock(return_value=SkillResult(skill=skill)),
    )


def _mock_qa_answer(response: str):
    return patch(
        "app.llm.response_generator.QaResponseGenerator.generate",
        AsyncMock(return_value=response),
    )


def _mock_course_search(courses: list[dict] | None = None):
    """Stub the vector search so Q&A tests don't need a live ChromaDB server."""
    if courses is None:
        courses = [
            {
                "id": "python-foundations",
                "name": "Python Foundations",
                "category": "programming",
                "level": "Beginner",
                "instructor": "Dr. Amara Okafor",
                "price": 199.00,
                "duration": "6 weeks",
                "seats_available": 24,
                "description": "Your first step into writing code.",
            }
        ]
    return patch("app.course.service.CourseService.search", return_value=courses)


def _mock_enrollment_draft():
    from app.llm.types import EnrollmentDraftResult

    result = EnrollmentDraftResult(
        course_id="python-foundations",
        course_name="Python Foundations",
        seats=1,
        total_price=199.00,
        note="",
    )
    return patch(
        "app.llm.response_generator.EnrollmentDraftGenerator.generate",
        AsyncMock(return_value=result),
    )


@pytest.mark.anyio
async def test_qa_course_search(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with (
        g1,
        g2,
        _mock_skill("qa"),
        _mock_course_search(),
        _mock_qa_answer("LLM: Python Foundations is a great starting point"),
    ):
        response = await client.post("/chat", json={"message": "Tell me about programming courses"})
    assert response.status_code == 200
    assert "Python Foundations" in response.json()["answer"]
    assert len(response.json()["sources"]) > 0


@pytest.mark.anyio
async def test_qa_no_match(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_skill("qa"), _mock_course_search([]), _mock_qa_answer("LLM: no courses found"):
        response = await client.post("/chat", json={"message": "Do you teach underwater basket weaving?"})
    assert response.status_code == 200
    assert "no courses" in response.json()["answer"].lower()


@pytest.mark.anyio
async def test_enroll_first_turn_prepare_draft(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_skill("enroll"), _mock_enrollment_draft():
        response = await client.post(
            "/chat", json={"message": "I want to enroll in Python Foundations", "session_id": "s1"}
        )
    assert response.status_code == 200
    assert "enrollment summary" in response.json()["answer"].lower()
    assert "confirm" in response.json()["answer"].lower()


@pytest.mark.anyio
async def test_enroll_second_turn_confirm(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_skill("enroll"), _mock_enrollment_draft():
        await client.post(
            "/chat", json={"message": "I want to enroll in Python Foundations", "session_id": "s2"}
        )
    # On resume, route_skill is skipped - no _mock_skill needed
    with g1, g2:
        response = await client.post("/chat", json={"message": "yes", "session_id": "s2"})
    assert response.status_code == 200
    assert "confirmed" in response.json()["answer"].lower()


@pytest.mark.anyio
async def test_enroll_second_turn_cancel(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_skill("enroll"), _mock_enrollment_draft():
        await client.post(
            "/chat", json={"message": "I want to enroll in Python Foundations", "session_id": "s3"}
        )
    # On resume, route_skill is skipped - no _mock_skill needed
    with g1, g2:
        response = await client.post("/chat", json={"message": "no", "session_id": "s3"})
    assert response.status_code == 200
    assert "cancelled" in response.json()["answer"].lower()


@pytest.mark.anyio
async def test_off_topic_rejected_by_guardrail(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=False)
    with g1, g2:
        response = await client.post("/chat", json={"message": "What is the weather today?"})
    assert response.status_code == 200
    assert "only help with questions about our courses" in response.json()["answer"]


@pytest.mark.anyio
async def test_empty_message_rejected_by_pydantic(client: AsyncClient) -> None:
    response = await client.post("/chat", json={"message": ""})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_session_id_preserved(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_skill("qa"), _mock_course_search(), _mock_qa_answer("LLM: hello"):
        response = await client.post("/chat", json={"message": "Hi", "session_id": "my-session-123"})
    assert response.status_code == 200
    assert response.json()["session_id"] == "my-session-123"


@pytest.mark.anyio
async def test_track_enrollment_no_enrollments(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_skill("track"):
        response = await client.post(
            "/chat", json={"message": "What's the status of my enrollment?", "session_id": "s9"}
        )
    assert response.status_code == 200
    assert "couldn't find any enrollments" in response.json()["answer"].lower()


@pytest.mark.anyio
async def test_track_enrollment_after_confirm(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    # Enroll in a course and confirm it
    with g1, g2, _mock_skill("enroll"), _mock_enrollment_draft():
        await client.post(
            "/chat", json={"message": "I want to enroll in Python Foundations", "session_id": "s10"}
        )
    # On resume, route_skill is skipped - no _mock_skill needed
    with g1, g2:
        await client.post("/chat", json={"message": "yes", "session_id": "s10"})

    # Now check its status
    with g1, g2, _mock_skill("track"):
        response = await client.post(
            "/chat", json={"message": "How is my course going?", "session_id": "s10"}
        )
    assert response.status_code == 200
    body = response.json()
    assert "ENR-1000" in body["answer"]
    assert "Python Foundations" in body["answer"]
    assert "confirmed" in body["answer"].lower()
    assert "not started" in body["answer"].lower()
