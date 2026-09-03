from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from app.agent.skills import AgentSkills
from app.agent.state import AgentState


class AgentGraph:
    """Builds and compiles the LangGraph agent graph.

    Receives AgentSkills via constructor — the graph wires skill methods
    as nodes and handles routing between them.
    """

    def __init__(self, skills: AgentSkills) -> None:
        self._skills = skills

    def build(self, checkpointer: BaseCheckpointSaver) -> StateGraph:
        """Build and compile the LangGraph agent graph.

        Requires a checkpointer for state persistence between turns.
        In production: SqliteSaver / PostgresSaver. In tests: InMemorySaver.

        Graph topology:

        Q&A path:
            route_skill ──(qa)──> search_courses ──> generate_qa_answer ──> END

        Enroll path (HITL via interrupt):
            route_skill ──(enroll)──> prepare_enrollment ──> await_confirmation
                                   drafts the enrollment       interrupt()
                        ──> finalize_enrollment ──> END
                             processes response

        Track path:
            route_skill ──(track)──> track_enrollment ──> END
        """
        graph: StateGraph = StateGraph(AgentState)

        graph.add_node("route_skill", self._skills.route_skill)
        graph.add_node("search_courses", self._skills.search_courses)
        graph.add_node("generate_qa_answer", self._skills.generate_qa_answer)
        graph.add_node("prepare_enrollment", self._skills.prepare_enrollment)
        graph.add_node("await_confirmation", self._skills.await_confirmation)
        graph.add_node("finalize_enrollment", self._skills.finalize_enrollment)
        graph.add_node("track_enrollment", self._skills.track_enrollment)

        graph.set_entry_point("route_skill")

        graph.add_conditional_edges(
            "route_skill",
            self._route_after_skill,
            {
                "search_courses": "search_courses",
                "prepare_enrollment": "prepare_enrollment",
                "track_enrollment": "track_enrollment",
            },
        )
        graph.add_edge("search_courses", "generate_qa_answer")
        graph.add_edge("generate_qa_answer", END)
        graph.add_edge("prepare_enrollment", "await_confirmation")
        graph.add_edge("await_confirmation", "finalize_enrollment")
        graph.add_edge("finalize_enrollment", END)
        graph.add_edge("track_enrollment", END)

        return graph.compile(checkpointer=checkpointer)

    @staticmethod
    def _route_after_skill(state: AgentState) -> str:
        """Conditional edge after skill routing.

        Q&A path: search courses → generate answer
        Enroll path: prepare enrollment draft → interrupt for HITL confirmation
        Track path: look up enrollment status in the registry
        """
        if state["skill"] == "enroll":
            return "prepare_enrollment"
        if state["skill"] == "track":
            return "track_enrollment"
        return "search_courses"
