from unittest.mock import patch

from app.course.catalog import courses_by_id, load_courses
from app.course.repository import ChromaCourseRepository
from app.course.service import CourseService

EXPECTED_COURSE_COUNT = 15


class _FakeCollection:
    """Stand-in for a ChromaDB collection — returns canned query results."""

    def __init__(self, ids: list[str], distances: list[float]) -> None:
        self._ids = ids
        self._distances = distances
        self.last_query: dict | None = None

    def query(self, query_texts, n_results, include):
        self.last_query = {"query_texts": query_texts, "n_results": n_results, "include": include}
        return {
            "ids": [self._ids[:n_results]],
            "distances": [self._distances[:n_results]],
        }


def _repo(**kwargs) -> ChromaCourseRepository:
    defaults = dict(host="x", port=1, collection_name="courses")
    defaults.update(kwargs)
    return ChromaCourseRepository(**defaults)


class TestCatalog:
    def test_load_courses_returns_full_catalog(self) -> None:
        assert len(load_courses()) == EXPECTED_COURSE_COUNT

    def test_every_course_has_required_fields(self) -> None:
        required = {
            "id",
            "name",
            "category",
            "level",
            "instructor",
            "price",
            "duration",
            "seats_available",
            "description",
        }
        for course in load_courses():
            assert required <= course.keys()

    def test_courses_by_id_indexes_catalog(self) -> None:
        by_id = courses_by_id()
        assert by_id["python-foundations"]["name"] == "Python Foundations"
        assert len(by_id) == EXPECTED_COURSE_COUNT


class TestChromaCourseRepository:
    def test_search_maps_ids_to_courses_in_order(self) -> None:
        repo = _repo(top_k=2)
        fake = _FakeCollection(
            ids=["python-foundations", "web-dev-fullstack", "data-analysis-pandas"],
            distances=[0.10, 0.20, 0.30],
        )
        with patch.object(ChromaCourseRepository, "_get_collection", return_value=fake):
            results = repo.search("beginner coding class")

        assert [c["id"] for c in results] == ["python-foundations", "web-dev-fullstack"]
        # top_k is forwarded to ChromaDB as n_results — retrieval, not a full scan.
        assert fake.last_query["n_results"] == 2

    def test_search_applies_max_distance_threshold(self) -> None:
        repo = _repo(top_k=3, max_distance=0.25)
        fake = _FakeCollection(
            ids=["python-foundations", "web-dev-fullstack", "data-analysis-pandas"],
            distances=[0.10, 0.20, 0.30],
        )
        with patch.object(ChromaCourseRepository, "_get_collection", return_value=fake):
            results = repo.search("programming")

        # The third hit (distance 0.30 > 0.25) is filtered out as not relevant enough.
        assert [c["id"] for c in results] == ["python-foundations", "web-dev-fullstack"]

    def test_search_skips_unknown_ids(self) -> None:
        repo = _repo(top_k=2)
        fake = _FakeCollection(ids=["python-foundations", "ghost-id"], distances=[0.1, 0.2])
        with patch.object(ChromaCourseRepository, "_get_collection", return_value=fake):
            results = repo.search("python")

        assert [c["id"] for c in results] == ["python-foundations"]

    def test_get_all_returns_full_catalog(self) -> None:
        assert len(_repo().get_all()) == EXPECTED_COURSE_COUNT


class TestCourseService:
    def test_search_delegates_to_repo(self) -> None:
        repo = _repo(top_k=1)
        fake = _FakeCollection(ids=["ux-design-fundamentals"], distances=[0.1])
        service = CourseService(repo)
        with patch.object(ChromaCourseRepository, "_get_collection", return_value=fake):
            results = service.search("design")
        assert results[0]["name"] == "UX Design Fundamentals"

    def test_get_all_delegates_to_repo(self) -> None:
        service = CourseService(_repo())
        assert len(service.get_all()) == EXPECTED_COURSE_COUNT
