from app.enrollment.repository import EnrollmentRepository


class EnrollmentService:
    """Enrollment creation and status lookup.

    Depends on EnrollmentRepository interface - never knows the concrete implementation.
    """

    def __init__(self, repo: EnrollmentRepository) -> None:
        self._repo = repo

    def create_enrollment(self, session_id: str, enrollment: dict) -> str:
        return self._repo.save(session_id, enrollment)

    def find_enrollments(self, session_id: str) -> list[dict]:
        return self._repo.find_by_session(session_id)
