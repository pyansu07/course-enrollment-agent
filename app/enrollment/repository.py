from abc import ABC, abstractmethod


class EnrollmentRepository(ABC):
    """Interface for enrollment persistence.

    In production: relational DB or a student information system API.
    """

    @abstractmethod
    def save(self, session_id: str, enrollment: dict) -> str:
        """Persist a confirmed enrollment and return its ID."""
        ...

    @abstractmethod
    def find_by_session(self, session_id: str) -> list[dict]:
        """Return all enrollments for a given session."""
        ...


class InMemoryEnrollmentRepository(EnrollmentRepository):
    """In-memory enrollment registry for development and testing."""

    def __init__(self) -> None:
        self._enrollments: dict[str, dict] = {}
        self._next_id = 1000

    def save(self, session_id: str, enrollment: dict) -> str:
        enrollment_id = f"ENR-{self._next_id}"
        self._next_id += 1
        self._enrollments[enrollment_id] = {
            "enrollment_id": enrollment_id,
            "session_id": session_id,
            "course_name": enrollment.get("course_name", ""),
            "seats": enrollment.get("seats", 0),
            "total_price": enrollment.get("total_price", 0),
            "status": "confirmed",
            "progress": "not started",
            "starts_in": "next cohort, 2 weeks",
        }
        return enrollment_id

    def find_by_session(self, session_id: str) -> list[dict]:
        return [e for e in self._enrollments.values() if e["session_id"] == session_id]

    def reset(self) -> None:
        """Clear all enrollments and reset the ID counter (for tests)."""
        self._enrollments.clear()
        self._next_id = 1000
