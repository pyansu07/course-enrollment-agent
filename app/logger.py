import logging
import sys


def setup_logger(name: str = "agent") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "\n%(levelname)s | %(message)s",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def format_state(state: dict) -> str:
    """Pretty-print agent state for logs - one message per line in chat history."""
    messages = state.get("messages", [])
    msg_lines = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        # Truncate long messages for readability
        preview = content[:200] + "..." if len(content) > 200 else content
        msg_lines.append(f"  [{role}] {preview}")

    parts = [
        f"session_id: {state.get('session_id', '?')}",
        f"skill: {state.get('skill') or '(none)'}",
        f"has_enrollment: {bool(state.get('enrollment'))}",
        f"enrollment_confirmed: {state.get('enrollment_confirmed', False)}",
    ]

    if state.get("enrollment"):
        enrollment = state["enrollment"]
        parts.append(
            f"enrollment: {enrollment.get('course_name')} x{enrollment.get('seats')} "
            f"= ${enrollment.get('total_price')}"
        )

    if state.get("course_results"):
        sources = state["course_results"]
        if isinstance(sources[0], dict):
            parts.append(f"course_results: {[c['name'] for c in sources]}")
        else:
            parts.append(f"course_results: {sources}")

    return (
        "\n  "
        + "\n  ".join(parts)
        + "\n\n  messages:\n"
        + ("\n".join(msg_lines) if msg_lines else "  (empty)")
        + f"\n\n  final_answer: {state.get('final_answer', '')[:200]}"
    )
