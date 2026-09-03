import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Must run before importing app.config.di — it reads os.environ["OPENAI_API_KEY"]
# at import time, so .env has to be loaded first when running outside Docker
# (where docker-compose injects env vars directly instead).
load_dotenv()

from fastapi import FastAPI  # noqa: E402
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: E402
from langgraph.graph import StateGraph  # noqa: E402
from langgraph.types import Command  # noqa: E402

from app.agent.state import AgentState  # noqa: E402
from app.config.di import agent_graph_builder, guardrail  # noqa: E402
from app.logger import format_state, setup_logger  # noqa: E402
from app.models import ChatRequest, ChatResponse  # noqa: E402

agent: StateGraph | None = None
logger = setup_logger("agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: manage AsyncSqliteSaver lifecycle."""
    global agent
    async with AsyncSqliteSaver.from_conn_string("data/checkpoints.db") as checkpointer:
        agent = agent_graph_builder.build(checkpointer=checkpointer)
        logger.info("Checkpointer ready (AsyncSqliteSaver: data/checkpoints.db)")
        yield
    logger.info("Checkpointer closed")


app = FastAPI(title="Course Enrollment Assistant", lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id: str = request.session_id or str(uuid.uuid4())
    config: dict = {"configurable": {"thread_id": session_id}}

    t_start = time.perf_counter()
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("REQUEST | session=%s | message=%s", session_id, request.message[:100])

    # Check if there's a pending interrupt for this session
    snapshot = await agent.aget_state(config)
    has_interrupt = bool(snapshot.next)
    history_messages = snapshot.values.get("messages") if snapshot.values else None

    # Input guardrail with conversation history for context
    t0 = time.perf_counter()
    input_check = await guardrail.check_input(request.message, history=history_messages)
    logger.info(
        "GUARD INPUT  [%.2fs] | on_topic=%s | reason=%s",
        time.perf_counter() - t0,
        input_check.on_topic,
        input_check.reason,
    )

    if not input_check.on_topic:
        logger.info("DONE [%.2fs] | rejected by input guardrail", time.perf_counter() - t_start)
        return ChatResponse(
            answer="I can only help with questions about our courses and your enrollments.",
            session_id=session_id,
            sources=[],
        )

    # Invoke graph - resume from interrupt or start new run
    t_graph = time.perf_counter()
    if has_interrupt:
        logger.info("GRAPH RESUME | resuming from interrupt")
        result = await agent.ainvoke(Command(resume=request.message), config)
    else:
        initial_state: AgentState = {
            "session_id": session_id,
            "messages": [{"role": "user", "content": request.message}],
            "skill": "",
            "course_results": [],
            "final_answer": "",
        }
        logger.info("GRAPH INPUT%s", format_state(initial_state))
        result = await agent.ainvoke(initial_state, config)

    logger.info(
        "GRAPH OUTPUT [%.2fs]%s",
        time.perf_counter() - t_graph,
        format_state(result),
    )

    # Output guardrail - LLM validates the response before returning to user
    t1 = time.perf_counter()
    output_check = await guardrail.check_output(result.get("final_answer", ""))
    logger.info(
        "GUARD OUTPUT [%.2fs] | valid=%s | reason=%s",
        time.perf_counter() - t1,
        output_check.valid,
        output_check.reason,
    )

    if not output_check.valid:
        logger.warning("GUARD OUTPUT rejected response: %s", output_check.reason)
        return ChatResponse(
            answer="I'm sorry, I ran into an issue processing your request. Could you try again or rephrase?",
            session_id=session_id,
            sources=result.get("course_results", []),
        )

    logger.info("DONE [%.2fs] | total request time", time.perf_counter() - t_start)

    return ChatResponse(
        answer=result["final_answer"],
        session_id=session_id,
        sources=result.get("course_results", []),
    )
