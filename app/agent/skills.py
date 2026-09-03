from langgraph.types import interrupt

from app.agent.state import AgentState
from app.course.service import CourseService
from app.enrollment.service import EnrollmentService
from app.llm.response_generator import EnrollmentDraftGenerator, QaResponseGenerator
from app.llm.skill_router import SkillRouter


class AgentSkills:
    """All agent skills wired together with constructor-based DI.

    Each skill is a method that receives AgentState and returns a partial
    state update dict — the standard LangGraph node signature.
    """

    def __init__(
        self,
        skill_router: SkillRouter,
        qa_generator: QaResponseGenerator,
        enrollment_generator: EnrollmentDraftGenerator,
        course_service: CourseService,
        enrollment_service: EnrollmentService,
    ) -> None:
        self._skill_router = skill_router
        self._qa_generator = qa_generator
        self._enrollment_generator = enrollment_generator
        self._course_service = course_service
        self._enrollment_service = enrollment_service

    async def route_skill(self, state: AgentState) -> dict:
        """Classify user intent: Q&A, Enroll, or Track.

        Only runs for new messages - on HITL resume, LangGraph skips
        the entry point and continues from the interrupt directly.
        """
        last_message: str = state["messages"][-1]["content"]
        result = await self._skill_router.classify(last_message)
        return {"skill": result.skill}

    def search_courses(self, state: AgentState) -> dict:
        """Search the course catalog via CourseService."""
        query: str = state["messages"][-1]["content"]
        results: list[dict[str, object]] = self._course_service.search(query)
        return {"course_results": results}

    async def generate_qa_answer(self, state: AgentState) -> dict:
        """Compose course Q&A response - LLM with search results as context (RAG generation)."""
        courses = state.get("course_results", [])
        user_message: str = state["messages"][-1]["content"]
        answer: str = await self._qa_generator.generate(user_message, courses)

        sources = [str(c["name"]) for c in courses] if courses else []
        return {
            "final_answer": answer,
            "messages": [*state["messages"], {"role": "assistant", "content": answer}],
            "course_results": sources,
        }

    async def prepare_enrollment(self, state: AgentState) -> dict:
        """Generate an enrollment draft via LLM and return it for HITL confirmation.

        The actual confirmation pause happens in await_confirmation node.
        """
        last_message: str = state["messages"][-1]["content"]
        all_courses: list[dict[str, object]] = self._course_service.get_all()

        draft_result = await self._enrollment_generator.generate(
            last_message, all_courses, history=state.get("messages", [])
        )
        draft: dict = {
            "course_id": draft_result.course_id,
            "course_name": draft_result.course_name,
            "seats": draft_result.seats,
            "total_price": draft_result.total_price,
        }

        answer = (
            f"Here's your enrollment summary:\n\n"
            f"Course: {draft['course_name']}\n"
            f"Seats: {draft['seats']}\n"
            f"Total: ${draft['total_price']:.2f}\n\n"
            f"Would you like to confirm this enrollment? (yes/no)"
        )

        return {
            "enrollment": draft,
            "enrollment_confirmed": False,
            "final_answer": answer,
            "messages": [*state["messages"], {"role": "assistant", "content": answer}],
        }

    def await_confirmation(self, state: AgentState) -> dict:
        """Pause execution and wait for user confirmation via LangGraph interrupt.

        On first pass: interrupt() pauses the graph, checkpointer persists state.
        On resume: interrupt() returns the value from Command(resume=...).
        """
        user_response: str = interrupt("Waiting for enrollment confirmation")

        return {
            "user_response": user_response,
            "messages": [
                *state["messages"],
                {"role": "user", "content": user_response},
            ],
        }

    def finalize_enrollment(self, state: AgentState) -> dict:
        """Process HITL confirmation and create or cancel the enrollment."""
        user_response: str = state.get("user_response", "")
        confirmed: bool = user_response.lower().strip() in (
            "yes",
            "yeah",
            "y",
            "confirm",
            "ok",
            "okay",
        )

        enrollment = state.get("enrollment", {})
        if confirmed:
            enrollment_id = self._enrollment_service.create_enrollment(state["session_id"], enrollment)
            answer = (
                f"Enrollment {enrollment_id} confirmed! You're booked onto "
                f"{enrollment.get('course_name', 'the course')}. "
                f"Course materials open 2 weeks before the cohort starts. "
                f"Total: ${enrollment.get('total_price', 0):.2f}"
            )
        else:
            answer = "Enrollment cancelled. Let me know if you need anything else."

        return {
            "enrollment_confirmed": confirmed,
            "final_answer": answer,
            "messages": [*state["messages"], {"role": "assistant", "content": answer}],
        }

    def track_enrollment(self, state: AgentState) -> dict:
        """Look up enrollments for the current session and return status and progress."""
        enrollments = self._enrollment_service.find_enrollments(state["session_id"])

        if not enrollments:
            answer = (
                "I couldn't find any enrollments for your session. "
                "Have you enrolled in a course yet?"
            )
        else:
            lines = []
            for e in enrollments:
                lines.append(
                    f"Enrollment {e['enrollment_id']}: {e['course_name']} x{e['seats']} - "
                    f"${e['total_price']:.2f} | Status: {e['status']} | "
                    f"Progress: {e['progress']} | Starts: {e['starts_in']}"
                )
            answer = "Here are your enrollments:\n\n" + "\n".join(lines)

        return {
            "final_answer": answer,
            "messages": [*state["messages"], {"role": "assistant", "content": answer}],
        }
