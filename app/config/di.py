"""Dependency injection — module-level singletons wired together.

One instance of each component, shared across all requests.
Tests import these directly and call enrollment_repo.reset() between cases.
"""

import os

from app.agent.graph import AgentGraph
from app.agent.skills import AgentSkills
from app.course.embeddings import DEFAULT_EMBEDDING_MODEL
from app.course.repository import ChromaCourseRepository
from app.course.service import CourseService
from app.enrollment.repository import InMemoryEnrollmentRepository
from app.enrollment.service import EnrollmentService
from app.llm.client import LlmClient
from app.llm.guardrail import Guardrail
from app.llm.response_generator import EnrollmentDraftGenerator, QaResponseGenerator
from app.llm.skill_router import SkillRouter

# Repositories (infrastructure)
# Course search is backed by ChromaDB (semantic retrieval). The collection is
# populated offline by the indexer (`make index`); the client connects lazily.
# Embeddings are computed locally — no API key, no per-query cost.
course_repo = ChromaCourseRepository(
    host=os.environ.get("CHROMA_HOST", "localhost"),
    port=int(os.environ.get("CHROMA_PORT", "8000")),
    collection_name=os.environ.get("CHROMA_COLLECTION", "courses"),
    embedding_model=os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
    top_k=int(os.environ.get("RAG_TOP_K", "3")),
)
enrollment_repo = InMemoryEnrollmentRepository()

# Domain services
course_service = CourseService(course_repo)
enrollment_service = EnrollmentService(enrollment_repo)

# LLM client — one ChatOpenAI shared across all LLM components.
# OPENAI_BASE_URL selects the provider (unset = OpenAI, Groq's /openai/v1 = Groq).
llm_client = LlmClient(
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ.get("OPENAI_MODEL", "openai/gpt-oss-120b"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

# LLM components — each receives the same ChatOpenAI instance
skill_router = SkillRouter(llm_client.chat_openai)
qa_generator = QaResponseGenerator(llm_client.chat_openai)
enrollment_generator = EnrollmentDraftGenerator(llm_client.chat_openai)
guardrail = Guardrail(llm_client.chat_openai)

# Agent skills — receives everything it needs via constructor
skills = AgentSkills(
    skill_router=skill_router,
    qa_generator=qa_generator,
    enrollment_generator=enrollment_generator,
    course_service=course_service,
    enrollment_service=enrollment_service,
)

# Agent graph builder — wires skills into the LangGraph graph
agent_graph_builder = AgentGraph(skills=skills)
