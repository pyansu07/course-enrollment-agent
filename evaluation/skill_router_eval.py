"""Frozen evaluation set for the LLM skill router.

The router is a prompt, so its accuracy can regress silently whenever the prompt or
the model changes. This replays a labelled set through the real classifier and reports
per-skill accuracy plus a confusion breakdown.

Hits the real chat API — one call per case.

Usage:
    python -m evaluation.skill_router_eval              # current (fixed) prompt
    python -m evaluation.skill_router_eval --baseline   # pre-fix prompt, reproduces the "before" number
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv

import app.llm.skill_router as skill_router_module
from app.llm.client import LlmClient
from app.llm.skill_router import SkillRouter

load_dotenv()

# Groq's free tier caps tokens-per-minute (8k TPM at time of writing), and a 54-case
# sweep of a few-shot prompt exceeds that comfortably. Retry on 429 rather than
# letting a rate limit look like a routing failure and corrupt the accuracy number.
_MAX_RETRIES = 6
_INITIAL_BACKOFF_SECONDS = 5.0


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "rate_limit" in text or "429" in text or "rate limit" in text


async def classify_with_retry(message: str):
    """Classify one message, backing off and retrying when rate limited.

    A rate limit is an infrastructure problem, not a wrong answer — counting it as a
    misclassification would silently understate accuracy.
    """
    backoff = _INITIAL_BACKOFF_SECONDS
    for attempt in range(_MAX_RETRIES):
        try:
            return await skill_router.classify(message)
        except Exception as exc:
            if not _is_rate_limit(exc) or attempt == _MAX_RETRIES - 1:
                raise
            print(f"  [rate limited, retrying in {backoff:.0f}s]")
            await asyncio.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unreachable")

# The router prompt as it stood before the manage-vs-track fix. Kept solely so the
# "before" number in the README stays reproducible — nothing in the app imports it.
# It defines each skill by topic rather than by action, which is precisely the defect:
# "cancel my enrollment" and "what's my status?" are both *about* an existing
# enrollment, so the classifier had no stated basis for separating them.
SKILL_ROUTER_PROMPT_V1: str = """You are a skill router for a course enrollment assistant.

Classify the user's message into one of three skills:
- "qa" - the user is asking a course question, browsing, or researching
- "enroll" - the user wants to enroll in, sign up for, or register for a course
- "track" - the user wants to check enrollment status, track course progress, or ask about an existing enrollment

Examples:
"I want to enroll in a Python course" → enroll
"Tell me about your data courses" → qa
"Do you have any design courses?" → qa
"Sign me up for that one" → enroll
"What's the price of UX Design Fundamentals?" → qa
"I'd like to register for Machine Learning Essentials" → enroll
"Where is my enrollment?" → track
"Track my course progress" → track
"What's the status of my enrollment?" → track
"Has my enrollment been confirmed yet?" → track"""

llm_client = LlmClient(
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ.get("OPENAI_MODEL", "openai/gpt-oss-120b"),
    base_url=os.environ.get("OPENAI_BASE_URL"),
)

skill_router = SkillRouter(llm_client.chat_openai)

SKILLS = ("qa", "enroll", "track")

messages_skills: list[tuple[str, str]] = [
    # Q&A — clear cases
    ("Which course do you recommend for learning data science?",   "qa"),
    ("How much does the UX design course cost?",                   "qa"),
    ("What topics are covered in Machine Learning Essentials?",    "qa"),
    ("Do you have any beginner design courses?",                   "qa"),
    ("What is the difference between the two security courses?",   "qa"),
    ("Which course is best for someone switching careers?",        "qa"),
    ("How long is the full-stack web development course?",         "qa"),
    ("Is Cloud Architecture on AWS worth taking?",                 "qa"),
    ("What courses do you have with seats still available?",       "qa"),
    ("Tell me about the Product Management course",                "qa"),
    ("What level is the deep learning course?",                    "qa"),
    ("Do you teach public speaking?",                              "qa"),
    ("What is the price range for your business courses?",         "qa"),
    ("Compare the beginner and advanced security courses",         "qa"),
    ("Does the marketing course cover paid advertising?",           "qa"),
    ("Who teaches the data analysis course?",                      "qa"),
    ("What is included in the financial modelling course?",        "qa"),

    # Enroll — clear cases
    ("I want to enroll in the Python course",                      "enroll"),
    ("Sign me up for a design course",                             "enroll"),
    ("I would like to register for the marketing course",          "enroll"),
    ("Book me a place on the DevOps course",                       "enroll"),
    ("Enroll me in the cheapest business course",                  "enroll"),
    ("I want to cancel my enrollment",                             "enroll"),
    ("I want to withdraw from my course",                          "enroll"),
    ("I want to enroll 2 people in the UX course",                 "enroll"),
    ("Can you register me for Cloud Architecture on AWS?",         "enroll"),
    ("I need to cancel my registration",                           "enroll"),
    ("Process a refund for my enrollment",                         "enroll"),
    ("I would like to switch to a different course",               "enroll"),
    ("Withdraw me from the course I signed up for",                "enroll"),
    ("Remove my enrollment",                                       "enroll"),
    ("I want to take the cybersecurity course",                    "enroll"),
    ("I want to cancel enrollment #123",                           "enroll"),
    ("Add the public speaking course to my enrollments",           "enroll"),

    # Track — clear cases
    ("How far along am I in my course?",                           "track"),
    ("When does enrollment #456 start?",                           "track"),
    ("Enrollment status",                                          "track"),
    ("What is the status of my enrollment?",                       "track"),
    ("Has my enrollment been confirmed yet?",                      "track"),
    ("Where is my course progress?",                               "track"),
    ("Track my enrollment",                                        "track"),
    ("Am I enrolled yet?",                                         "track"),
    ("My course has not started yet",                              "track"),
    ("Check my enrollment status",                                 "track"),
    ("When does my cohort begin?",                                 "track"),
    ("Track enrollment #789",                                      "track"),
    ("What is the start date for my course?",                      "track"),

    # Edge cases — easy to confuse
    ("I want to drop the course, what is my status?",              "enroll"),
    ("Cancel and refund my enrollment",                            "enroll"),
    ("Where is the course I signed up for?",                       "track"),
    ("Did my enrollment go through?",                              "track"),
    ("I am not happy with the course, I want a refund",            "enroll"),
    ("My cohort is late starting, can you check?",                 "track"),
    ("I changed my mind, cancel everything",                       "enroll"),
]


async def evaluate(baseline: bool = False, delay: float = 0.0):
    if baseline:
        # classify() reads this module global at call time, so rebinding it swaps the prompt.
        skill_router_module.SKILL_ROUTER_PROMPT = SKILL_ROUTER_PROMPT_V1

    label = "BASELINE (pre-fix prompt)" if baseline else "CURRENT (fixed prompt)"
    model = os.environ.get("OPENAI_MODEL", "openai/gpt-oss-120b")
    print(f"Skill router evaluation — {label}")
    print(f"Model: {model} | {len(messages_skills)} cases\n")

    wrong_predictions: dict[str, int] = {}
    per_skill: dict[str, dict[str, int]] = {s: {"total": 0, "correct": 0} for s in SKILLS}
    failures: list[tuple[str, str, str]] = []

    correct = 0
    for index, (message, expected_skill) in enumerate(messages_skills):
        # Pace the sweep to stay under the provider's tokens-per-minute ceiling.
        if delay and index:
            await asyncio.sleep(delay)

        result = await classify_with_retry(message)
        result_correct = result.skill == expected_skill

        per_skill.setdefault(expected_skill, {"total": 0, "correct": 0})
        per_skill[expected_skill]["total"] += 1

        if result_correct:
            correct += 1
            per_skill[expected_skill]["correct"] += 1
        else:
            key = expected_skill + ":" + result.skill
            wrong_predictions[key] = wrong_predictions.get(key, 0) + 1
            failures.append((message, expected_skill, result.skill))

        status = "CORRECT" if result_correct else "WRONG"
        print(f'Message: "{message}", expected: {expected_skill}, router result: {result.skill} - {status}')

    total = len(messages_skills)

    print("\n" + "=" * 62)
    print(f"{'Skill':<10} {'Cases':>7} {'Correct':>9} {'Accuracy':>12}")
    print("-" * 62)
    for skill in SKILLS:
        stats = per_skill.get(skill, {"total": 0, "correct": 0})
        if not stats["total"]:
            continue
        acc = stats["correct"] / stats["total"]
        print(f"{skill:<10} {stats['total']:>7} {stats['correct']:>9} {acc:>11.2%}")
    print("-" * 62)
    print(f"{'TOTAL':<10} {total:>7} {correct:>9} {correct / total:>11.2%}")
    print("=" * 62)

    if wrong_predictions:
        print("\nConfusion breakdown:")
        for key, count in sorted(wrong_predictions.items(), key=lambda kv: -kv[1]):
            expected_skill, result_skill = key.split(":")
            print(f"  expected={expected_skill}, got={result_skill} x {count}")

        print("\nFailing cases:")
        for message, expected_skill, got_skill in failures:
            print(f'  "{message}"  expected={expected_skill}  got={got_skill}')
    else:
        print("\nNo misclassifications.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the LLM skill router.")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="run against the pre-fix router prompt to reproduce the 'before' number",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=float(os.environ.get("EVAL_DELAY_SECONDS", "1.5")),
        help="seconds to wait between cases, to stay under free-tier rate limits",
    )
    args = parser.parse_args()
    asyncio.run(evaluate(baseline=args.baseline, delay=args.delay))
