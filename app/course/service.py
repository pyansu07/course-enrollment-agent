from app.course.repository import CourseRepository


class CourseService:
    """Course search and retrieval.

    Depends on CourseRepository interface - never knows the concrete implementation.
    """

    def __init__(self, repo: CourseRepository) -> None:
        self._repo = repo

    def search(self, query: str) -> list[dict]:
        return self._repo.search(query)

    def get_all(self) -> list[dict]:
        return self._repo.get_all()
