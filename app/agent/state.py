from typing import NotRequired, TypedDict


class EnrollmentDraft(TypedDict):
    course_id: str
    course_name: str
    seats: int
    total_price: float


class AgentState(TypedDict):
    """State that flows through LangGraph nodes.

    TypedDict is a lightweight alternative to Pydantic for graph state.
    LangGraph copies this dict between nodes - each node returns a partial
    update that gets merged into the accumulated state.
    """

    session_id: str
    messages: list[dict[str, str]]
    skill: str
    course_results: list[dict[str, object]]
    enrollment: NotRequired[EnrollmentDraft]
    enrollment_confirmed: NotRequired[bool]
    user_response: NotRequired[str]
    final_answer: str
