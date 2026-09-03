# Architecture

A single-endpoint FastAPI service that wraps a LangGraph state machine. One `POST /chat`
call goes through two LLM guardrails, an LLM intent classifier, and one of three paths
through a 7-node graph. State survives between HTTP calls via a SQLite checkpointer,
which is what makes the human-in-the-loop confirmation step possible.

If you read only one section, read [The 90-second version](#the-90-second-version) and
[HITL: how the pause actually works](#hitl-how-the-pause-actually-works). Those two carry
the parts that are actually interesting.

---

## The 90-second version

```
POST /chat {message, session_id}
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │ main.py                                             │
  │  1. aget_state(config) ── is a run already paused?  │
  │  2. input guardrail     (LLM call)                  │
  │  3. invoke graph        (new run OR resume)         │
  │  4. output guardrail    (LLM call)                  │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  route_skill  ──(qa)──────▶ search_courses ─▶ generate_qa_answer ─▶ END
       │
       ├───────(enroll)──▶ prepare_enrollment ─▶ await_confirmation ─▶ finalize_enrollment ─▶ END
       │                                              ⏸ interrupt()
       │
       └───────(track)───▶ track_enrollment ────────────────────────────────────────────────▶ END
```

Three things to notice, because they're the design decisions worth defending:

1. **The `session_id` is the graph's `thread_id`.** That one mapping is what gives the
   service memory. There is no session table, no cache — LangGraph's checkpointer is the
   session store.
2. **Retrieval is a node, not middleware.** Only the `qa` path touches the vector store.
   An "enroll me in X" or "what's my status" request never pays for an embedding call.
   This is what "agentic RAG" means here: the agent decides whether to retrieve.
3. **The pause is a real suspension, not a flag.** `await_confirmation` calls LangGraph's
   `interrupt()`, which unwinds graph execution and persists a checkpoint. The next HTTP
   request resumes the graph mid-node. No `awaiting_confirmation = True` column anywhere.

---

## Request flow, end to end

Everything below happens in [`app/main.py`](app/main.py) in the `chat` handler.

### 1. Resolve the session

```python
session_id = request.session_id or str(uuid.uuid4())
config = {"configurable": {"thread_id": session_id}}
```

`thread_id` is the key LangGraph uses to look up prior checkpoints. Same `session_id` →
same conversation state.

### 2. Check for a pending interrupt

```python
snapshot = await agent.aget_state(config)
has_interrupt = bool(snapshot.next)
```

`snapshot.next` is the tuple of nodes the graph would run next. For a finished or
brand-new thread it's empty `()`. If a previous call hit `interrupt()`, it's
`("await_confirmation",)`. **This single boolean decides whether the incoming message is
a new request or an answer to a question the agent already asked.** That's the whole
routing mechanism for the HITL turn — there is no separate state machine tracking it.

The snapshot also yields `messages`, which is fed to the input guardrail as context so a
bare `"yes"` isn't judged in isolation.

### 3. Input guardrail (LLM call)

[`Guardrail.check_input`](app/llm/guardrail.py) classifies the message into
`InputGuardResult{on_topic, reason}` via structured output. The prompt is deliberately
lenient — it allows by default and blocks only explicit categories (harassment, doing the
user's coursework for them, sexual content, illegal requests, prompt-injection attempts).
A blocked message returns a canned refusal and never reaches the graph.

Note the domain-specific carve-out: this platform *teaches programming*, so unlike a
generic assistant the guard must not block coding questions. What it blocks instead is
"write my assignment for me".

### 4. Invoke the graph — new run or resume

```python
if has_interrupt:
    result = await agent.ainvoke(Command(resume=request.message), config)
else:
    result = await agent.ainvoke(initial_state, config)
```

Two genuinely different invocations. `Command(resume=...)` does **not** re-enter at the
entry point — it re-enters *inside* `await_confirmation`, at the exact `interrupt()` call
that suspended. This is why `route_skill` does not run on a confirmation turn, and why the
tests for the confirm/cancel turns don't need to mock the skill router.

### 5. Output guardrail (LLM call)

[`Guardrail.check_output`](app/llm/guardrail.py) validates `final_answer` for obvious
breakage (empty, placeholder text, tracebacks). Failures are swapped for an apology
string. Note this fires on *every* response, including the draft-confirmation prompt.

### 6. Respond

`ChatResponse{answer, session_id, sources}`. `sources` carries the retrieved course names
for the Q&A path, empty otherwise.

**LLM calls per request:** 4 on a Q&A turn (input guard, router, answer generation, output
guard), 4 on a draft turn, 2 on a confirm turn (guards only — resume skips the router, and
`finalize_enrollment` is pure Python), 3 on a track turn.

---

## The 7 nodes

Registered in [`app/agent/graph.py`](app/agent/graph.py); all implemented as methods on
[`AgentSkills`](app/agent/skills.py). Every node takes `AgentState` and returns a **partial**
dict that LangGraph merges into accumulated state.

| # | Node | Async | LLM | Returns |
|---|------|-------|-----|---------|
| 1 | `route_skill` | ✅ | ✅ | `{skill}` |
| 2 | `search_courses` | — | — | `{course_results}` |
| 3 | `generate_qa_answer` | ✅ | ✅ | `{final_answer, messages, course_results}` |
| 4 | `prepare_enrollment` | ✅ | ✅ | `{enrollment, enrollment_confirmed, final_answer, messages}` |
| 5 | `await_confirmation` | — | — | `{user_response, messages}` |
| 6 | `finalize_enrollment` | — | — | `{enrollment_confirmed, final_answer, messages}` |
| 7 | `track_enrollment` | — | — | `{final_answer, messages}` |

### 1. `route_skill` — the only branch point

Takes the last message, calls [`SkillRouter.classify`](app/llm/skill_router.py), writes
`skill` into state. The conditional edge `_route_after_skill` then reads that string and
picks the path. Note the fallthrough: `enroll` and `track` are explicit, **everything else
falls through to `search_courses`** — so a malformed classification degrades to Q&A rather
than erroring.

### 2. `search_courses` — RAG retrieval

The only node that touches the vector store, and the only sync node on the Q&A path.
Delegates to `CourseService.search` → `ChromaCourseRepository.search`: embed the query
locally, ask Chroma for top-k by cosine distance, map returned ids back to the local
catalog. Chroma stores *vectors and ids*; the canonical records stay in the JSON catalog.
An optional `max_distance` cutoff can drop weak hits (unset by default, so Q&A always has
some context to work with).

### 3. `generate_qa_answer` — RAG generation

Formats the retrieved records into a context block (name, level, duration, instructor,
price, seats, description), injects it into `QA_ANSWER_PROMPT`, and asks the LLM to compose
an answer grounded in that data. Also flattens `course_results` from full dicts down to a
list of names, which is what the API returns as `sources`.

If retrieval returned nothing, it short-circuits with a fixed string and never calls the
LLM.

### 4. `prepare_enrollment` — LLM extraction into a draft

Calls [`EnrollmentDraftGenerator`](app/llm/response_generator.py) with the message, the
**full** catalog (not the retrieved subset), and the last 6 messages of history. Structured
output guarantees an `EnrollmentDraftResult{course_id, course_name, seats, total_price}`.

History matters here: it's what lets *"sign me up for that one"* resolve against a course
discussed two turns earlier. The node then writes the draft to state and composes the
"confirm this? (yes/no)" summary as `final_answer`.

Worth being honest about in an interview: the LLM computes `total_price` itself rather
than the code multiplying `price × seats`. That's a soft spot — arithmetic in the model
instead of in the domain layer.

### 5. `await_confirmation` — the suspension point

```python
user_response = interrupt("Waiting for enrollment confirmation")
```

This one line has two completely different behaviours:

- **First pass:** raises internally, unwinding graph execution. LangGraph persists a
  checkpoint marking `await_confirmation` as next. `ainvoke` returns; `main.py` sends the
  draft summary to the user.
- **On resume:** `interrupt()` *returns* the value from `Command(resume=...)` — the user's
  literal next message — and execution continues into the return statement as if it had
  never stopped.

### 6. `finalize_enrollment` — commit or discard

Pure Python, no LLM. Normalizes `user_response` against an affirmative set
(`yes/yeah/y/confirm/ok/okay`). On a match: `EnrollmentService.create_enrollment` persists
the record and returns an id, and the node composes a confirmation message. Anything else
is treated as a decline. **Fail-closed: ambiguous input never commits.**

### 7. `track_enrollment` — registry lookup

No LLM, no vector store. `EnrollmentService.find_enrollments(session_id)` and format,
reporting status, progress and cohort start. The session-scoped lookup is why status checks
only ever see enrollments from the current conversation.

---

## The skill router

[`app/llm/skill_router.py`](app/llm/skill_router.py) — about 25 lines, and all the
behaviour lives in the prompt.

```python
self._llm = llm.with_structured_output(SkillResult)   # SkillResult{skill: str}
```

`with_structured_output` binds a Pydantic schema to the model call, so the response is a
validated object rather than a string to parse. No regex, no JSON repair, no retry loop.
This is the one hard requirement on the model: it must support `json_schema` structured
output. Four components depend on it (router, draft extractor, both guardrails).

The classification itself is few-shot: [`SKILL_ROUTER_PROMPT`](app/llm/prompts.py) defines
each of the three skills plus a block of labelled examples. There is no keyword matching
anywhere in the path — swapping the classifier means editing a prompt, not editing the
graph.

**Why a separate LLM call instead of tool-calling?** Routing is a cheap, cacheable
classification with a fixed output space of three values. Making it its own step means it
can be evaluated in isolation — which is exactly what
[`evaluation/skill_router_eval.py`](evaluation/skill_router_eval.py) does, replaying a
frozen labelled set and reporting per-class accuracy plus a confusion breakdown.

### The bug this design invited

Because classification rests entirely on the prompt's wording, two intents that share
surface vocabulary but differ in *action* collide unless the prompt separates them
explicitly. The original prompt defined the skills by **topic**:

> `"track"` — the user wants to check enrollment status, track course progress, **or ask
> about an existing enrollment**

"Cancel my enrollment" *is* about an existing enrollment. So is "what's my status?". The
prompt gave the classifier no stated basis for telling them apart, and it collapsed all
enrollment-management intent into `track` — measured at **11 of 21 enroll cases failing**.

The fix reframes the definitions around **action rather than topic**, adds an explicit
precedence rule ("if the user wants something to *change*, choose enroll; if they only
want to *be told* something, choose track"), and supplies few-shot examples for the
cancel/withdraw/refund cases including a deliberately mixed one. See the README for the
before/after numbers.

---

## HITL: how the pause actually works

The part most worth being able to draw on a whiteboard.

```
TURN 1                                         TURN 2
──────                                         ──────
POST "I want to enroll in Python Foundations"  POST "yes"
     │                                              │
     ▼                                              ▼
aget_state → next=()                           aget_state → next=("await_confirmation",)
     │  new run                                     │  resume
     ▼                                              ▼
route_skill → "enroll"                         ainvoke(Command(resume="yes"))
     ▼                                              │
prepare_enrollment → draft in state                 ▼
     ▼                                         interrupt() RETURNS "yes"
await_confirmation                                  ▼
     ▼                                         finalize_enrollment → create_enrollment()
interrupt() ── unwinds ──┐                          ▼
                         ▼                     END
              checkpoint saved to SQLite            │
                         │                          ▼
                         ▼                "Enrollment ENR-1000 confirmed!"
   "Confirm? Python Foundations — $199.00"
```

The properties that fall out of this:

- **The draft is never persisted to the domain store.** It lives only in graph state until
  `finalize_enrollment` commits it. An abandoned confirmation leaves no enrollment behind.
- **It survives a restart.** The checkpointer is `AsyncSqliteSaver` over
  `data/checkpoints.db`, opened for the app's lifetime in the `lifespan` context manager.
  Kill the server mid-confirmation, restart, reply `"yes"` — it resumes. Tests swap in
  `InMemorySaver`, which is the entire reason `AgentGraph.build()` takes the checkpointer
  as a parameter instead of constructing one.
- **The user's reply is free-form.** It's whatever they typed, interpreted in
  `finalize_enrollment`. Nothing constrains it to yes/no at the transport layer.
- **One interrupt per thread.** Because the resume decision is driven by
  `bool(snapshot.next)`, a paused thread routes *every* incoming message to the resume
  branch until the graph reaches `END`. A user who changes the subject mid-confirmation has
  their new message consumed as the confirmation answer. Known sharp edge.

---

## Supporting structure

### State

[`AgentState`](app/agent/state.py) is a `TypedDict`, not a Pydantic model — LangGraph merges
partial dicts between nodes, and a plain dict keeps that cheap. `enrollment`,
`enrollment_confirmed` and `user_response` are `NotRequired`: they only exist on the enroll
path.

Note that `messages` uses **explicit list rebuilding** (`[*state["messages"], new_msg]`)
rather than an `Annotated[..., add_messages]` reducer. More verbose, but the append is
visible at each call site.

### Dependency injection

[`app/config/di.py`](app/config/di.py) is the composition root: module-level singletons,
constructor injection, wired bottom-up (repositories → services → LLM components → skills →
graph builder). No framework. `AgentSkills` receives five collaborators and knows none of
their concrete types.

The one thing to know: **it reads `os.environ["OPENAI_API_KEY"]` at import time**, so tests
must set that env var before importing anything from `app`.

### Package-by-feature

`course/` and `enrollment/` each own an ABC repository plus a service. `CourseRepository`
has a ChromaDB implementation; `EnrollmentRepository` has an in-memory one. The graph
depends only on the service layer, so replacing the in-memory registry with a real database
touches one class.

`chromadb` is imported **inside** `_get_collection()`, not at module top. That keeps import
of the DI container free of network I/O and lets unit tests run with no Chroma server up.
The embedding model is loaded lazily for the same reason — importing the DI container must
not pull in torch.

### Provider independence

The chat model is reached through [`LlmClient`](app/llm/client.py), which is just
`ChatOpenAI` with an optional `base_url`. Because Groq (and most others) expose an
OpenAI-compatible endpoint, switching providers is one environment variable and touches no
application code. The only compatibility requirement is `json_schema` structured output.

Embeddings are fully local — [`LocalHuggingFaceEmbeddingFunction`](app/course/embeddings.py)
wraps langchain's `HuggingFaceEmbeddings` behind Chroma's embedding-function interface, so
the indexer and repository keep their original shape while the vectors are computed on CPU
with no API call.

### Indexing vs. retrieval

Two phases, one collection, one embedding model:

- **Offline** — [`embedding/index.py`](embedding/index.py) (`make index`) flattens each
  catalog record to text, embeds it, upserts into Chroma. Drops and recreates the
  collection so re-runs are idempotent. Deliberately excludes price and seat count from the
  embedded text — numbers are noise for similarity search.
- **Online** — `search_courses` embeds the query with the same model and asks for top-k.

Both sides *must* use the same embedding model. Chroma fixes a collection's vector width at
creation time, so changing `EMBEDDING_MODEL` **requires** a re-index — which is why the
indexer always drops and recreates rather than upserting into whatever is already there.
The current model produces 384-dimensional vectors; the indexer prints and checks that
width on every run so a silent mismatch becomes a loud one.

---

## Extension points

- **New skill** → a method on `AgentSkills`, an `add_node`, a branch in
  `_route_after_skill`, and an entry in the router prompt. Existing paths untouched.
- **New HITL pause** → an `interrupt()` node anywhere; `main.py` already handles any
  paused thread generically.
- **Real persistence** → implement `EnrollmentRepository`, rebind in `di.py`.
- **Different vector store** → implement `CourseRepository`, rebind in `di.py`.
- **Different LLM provider** → set `OPENAI_BASE_URL` and `OPENAI_MODEL`. No code change.
