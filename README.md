# Course Enrollment Assistant — Agentic RAG + HITL

This is a multi-skill AI agent for a course platform, built on **FastAPI**, **LangGraph**,
and **ChromaDB**. It routes each incoming message to the right skill — course Q&A (RAG),
enrollment (with a human-in-the-loop confirmation step), or enrollment status — and keeps
the conversation state across turns so a user can say "yes" three messages later and have
it mean something. 7 graph nodes, 3 skills, 1 checkpointer.

The full breakdown of how it's wired — every node, the router, the interrupt/resume
mechanism — is in [ARCHITECTURE.md](ARCHITECTURE.md). This README covers what it does, a
router bug I found and fixed with measured numbers, and what I'd build next.

## What it does

Three skills, one endpoint (`POST /chat`), an LLM classifier deciding which path a message
takes:

| Skill | Trigger | What happens |
|-------|---------|-------------|
| **Q&A (RAG)** | "What courses do you have on data?" | Classify → semantic search the catalog → LLM composes an answer from the hits |
| **Enroll (HITL)** | "Sign me up for Python Foundations" | LLM extracts the course → draft summary → **pause and wait for confirmation** → commit |
| **Status** | "What's the status of my enrollment?" | Look up this session's enrollments → report status, progress, cohort start |

The HITL step is the part I think is worth calling out. It's not a "confirm: yes/no"
flag checked in application code — it's LangGraph's `interrupt()` primitive, which
actually suspends graph execution mid-node and persists the suspension to SQLite. The
server can restart between the draft and the confirmation and it still resumes correctly.
[ARCHITECTURE.md](ARCHITECTURE.md#hitl-how-the-pause-actually-works) walks through both
turns with a diagram.

Every request also passes through an LLM input guardrail (rejects off-topic/abusive
messages before they reach the agent) and an LLM output guardrail (catches obviously
broken responses before they reach the user).

## The bug I found and fixed

The skill router is a single LLM call classifying each message into `qa`, `enroll`, or
`track`, guided entirely by a prompt with a handful of labelled examples. Running it
against a frozen 54-case evaluation set turned up a real problem: **enrollment-management
messages — cancel, withdraw, refund — were being classified as status checks instead of
as enrollment actions.**

The original prompt defined the `track` skill this way:

> "the user wants to check enrollment status, track course progress, **or ask about an
> existing enrollment**"

"I want to cancel my enrollment" *is* about an existing enrollment. So the classifier had
no basis in the prompt for separating "tell me about my enrollment" from "change my
enrollment" — both are topically about the same thing. It collapsed almost every
management intent into `track`.

**Fix:** rewrote the prompt to classify by *action* instead of *topic* — an explicit rule
that says "if the user wants something changed, that's `enroll`; if they only want to be
told something, that's `track`" — and added few-shot examples for the specific cancel/
withdraw/refund phrasings the eval caught, including a deliberately ambiguous one ("I want
to drop the course, what is my status?") to pin down the precedence.

### Before / after (measured, not estimated)

Both numbers come from the same 54-case eval set, same model
(`openai/gpt-oss-120b` via Groq), one run each — reproducible with
`make eval-baseline` and `make eval`:

| Skill | Cases | Before | After |
|-------|-------|--------|-------|
| Q&A | 17 | 100.00% | 100.00% |
| Enroll | 21 | **47.62%** | **100.00%** |
| Track | 16 | 100.00% | 100.00% |
| **Total** | **54** | **79.63%** | **100.00%** |

11 of the 21 enroll cases were misrouted before the fix — every single one a
cancel/withdraw/refund message landing on `track`. All 11 pass now, and nothing else moved.

```
"I want to cancel my enrollment"                        expected=enroll  got=track  (before)
"Process a refund for my enrollment"                     expected=enroll  got=track  (before)
"I am not happy with the course, I want a refund"        expected=enroll  got=track  (before)
"I want to drop the course, what is my status?"          expected=enroll  got=track  (before)
...
```

I verified this isn't just an eval artifact — I hit the running `/chat` endpoint directly
with "I want to cancel my enrollment" and "I am not happy with the course, I want a
refund" and confirmed both now route to the enroll path.

**One thing this fix does *not* do:** it only fixes classification. The enroll skill
itself still only knows how to draft a *new* sign-up — it has no cancellation flow. Now
that cancel/refund messages correctly reach that node, they get a nonsense draft (no
course, $0.00) instead of a routing error. That's a real, separate gap — see below.

## Architecture at a glance

```
POST /chat → input guardrail → route_skill (LLM) ─┬─(qa)────→ search_courses → generate_qa_answer
                                                     ├─(enroll)→ prepare_enrollment → await_confirmation ⏸ → finalize_enrollment
                                                     └─(track)─→ track_enrollment
                                                                        ↓
                                                              output guardrail → response
```

`session_id` doubles as the LangGraph `thread_id` — that single mapping is the entire
session store, no separate cache or session table. Full detail, including the exact
mechanics of the `interrupt()` / `Command(resume=...)` pause, is in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Stack

| Component | Technology |
|-----------|-----------|
| API framework | FastAPI (async) |
| Agent orchestration | LangGraph (state graph, 7 nodes, 3 conditional paths) |
| LLM | `openai/gpt-oss-120b` via **Groq** (OpenAI-compatible endpoint, `langchain-openai`) — swappable to any compatible provider via one env var |
| RAG retrieval | ChromaDB 1.5.9 (separate container) |
| Embeddings | **Local** `sentence-transformers/all-MiniLM-L6-v2` (384-dim, CPU, no API key) |
| Data validation | Pydantic |
| State persistence | LangGraph SQLite checkpointer (+ InMemorySaver for tests) |
| Linting & formatting | ruff |
| Containerization | Docker + docker-compose |

Cost-wise, nothing in the request path calls a paid API: chat inference runs on Groq's
free tier and every embedding is computed locally. The only spend is whatever tokens the
free tier's rate limits allow you to burn.

### On the provider swap

The chat client is a thin wrapper around `ChatOpenAI` with an optional `base_url`
(`app/llm/client.py`) — Groq (and most inference providers now) expose an
OpenAI-compatible chat endpoint, so switching providers is `OPENAI_BASE_URL` +
`OPENAI_MODEL` in `.env`, no application code touched. The one real constraint: the model
has to support `json_schema` structured output, because four components
(`SkillRouter`, `EnrollmentDraftGenerator`, both guardrails) rely on
`with_structured_output()` for guaranteed-shape JSON. Not every Groq model does — I
verified `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, and `qwen/qwen3.8-27b` all work;
the `groq/compound*` models reject structured output requests outright.

Embeddings moved from OpenAI's `text-embedding-3-small` (1536-dim) to a local MiniLM model
(384-dim). ChromaDB fixes a collection's vector width at creation time, so this isn't a
drop-in swap — `embedding/index.py` drops and rebuilds the collection every time it runs,
which is exactly what makes changing `EMBEDDING_MODEL` safe rather than a silent
dimension-mismatch failure later.

## Quick start

```bash
# 1. Configure — copy the example and fill in your Groq key
cp .env.example .env

# 2. Bring up ChromaDB
docker compose up -d chroma

# 3. Install deps and build the vector index (first run downloads the embedding model, ~90MB)
pip install fastapi "uvicorn[standard]" pydantic langgraph langgraph-checkpoint-sqlite \
  aiosqlite langchain-openai python-dotenv "chromadb>=1.5.9,<2.0.0" \
  langchain-huggingface sentence-transformers
make index

# 4. Run
uvicorn app.main:app --reload
```

API: `http://localhost:8000` | Swagger: `http://localhost:8000/docs`

Or the fully containerized path: `make build && make up && make index`.

## Commands

```
make up                 start app + chroma with hot reload
make index              build the vector index (embed catalog into ChromaDB)
make inspect-documents  list stored docs + metadata in ChromaDB
make inspect-embeddings same, plus a preview of each stored vector
make down               stop services
make logs               tail logs
make test               run tests locally
make lint               ruff check
make format             ruff format
make eval               run the skill router eval (current prompt)
make eval-baseline      run the eval against the pre-fix prompt (reproduces the "before" numbers above)
```

## API

```bash
# Course Q&A
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What courses do you have on data?", "session_id": "s1"}'

# Enroll (step 1 - draft)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to enroll in Python Foundations", "session_id": "s2"}'

# Confirm enrollment (step 2)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "yes", "session_id": "s2"}'

# Check enrollment status
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the status of my enrollment?", "session_id": "s2"}'
```

Or use `api-requests.http`, which includes a set of semantic-RAG demo queries designed to
have zero keyword overlap with the course they should match (e.g. "our releases keep
breaking, everything is manual" → DevOps and CI/CD Pipelines) — a way to demonstrate real
vector retrieval isn't just matching on shared words.

## Testing

27 tests (`make test`): 10 integration tests against the HTTP layer with the LLM and
vector search mocked, plus unit tests for the course catalog/repository and the
enrollment repository/service. None of them hit a real LLM or a live ChromaDB — the eval
suite above is the one thing that does.

## What I'd improve next

- **No cancellation flow.** The router fix correctly sends "cancel my enrollment" to the
  `enroll` skill now, but that skill only knows how to draft a *new* sign-up — it has no
  code path for "find my existing enrollment and cancel it." Right now that produces an
  empty, nonsense draft rather than a routing error, which is arguably worse. This needs
  its own node (or a branch inside `prepare_enrollment`) that looks up the session's
  existing enrollments before deciding whether it's drafting a join or a cancellation.
- **Arithmetic lives in the LLM.** `EnrollmentDraftGenerator` asks the model to compute
  `total_price` itself instead of the code multiplying `price × seats`. Works fine on a
  flat-rate catalog with small numbers; I wouldn't trust it once seat-based discounts or
  bulk pricing exist.
- **One interrupt per thread, no escape hatch.** Once a thread is paused on
  `await_confirmation`, every message is consumed as the confirmation answer — a user who
  changes the subject mid-confirmation gets no path back to it until they answer yes/no.
  A cancel keyword that breaks out of the pending confirmation would fix this.
- **Rate limits make the eval slower than it should be.** Groq's free tier caps at 8k
  TPM, which the 54-case sweep exceeds with the fixed (longer) prompt. The eval now
  paces itself and retries on 429 rather than miscounting a rate limit as a wrong
  answer, but a paid tier would make `make eval` meaningfully faster.
- **Embedding model choice was pragmatic, not tuned.** MiniLM is fast and free, and the
  5 semantic-retrieval probes I ran by hand all landed the right course at rank 1 — but I
  didn't build a retrieval-quality eval the way the router has one. That's the obvious
  next eval to add.

