from app.enrollment.repository import InMemoryEnrollmentRepository
from app.enrollment.service import EnrollmentService


class TestInMemoryEnrollmentRepository:
    def test_save_returns_enrollment_id(self) -> None:
        repo = InMemoryEnrollmentRepository()
        enrollment_id = repo.save(
            "s1", {"course_name": "Python Foundations", "seats": 1, "total_price": 199.00}
        )
        assert enrollment_id == "ENR-1000"

    def test_save_increments_ids(self) -> None:
        repo = InMemoryEnrollmentRepository()
        id1 = repo.save("s1", {})
        id2 = repo.save("s1", {})
        assert id1 == "ENR-1000"
        assert id2 == "ENR-1001"

    def test_find_by_session_returns_matching_enrollments(self) -> None:
        repo = InMemoryEnrollmentRepository()
        repo.save("s1", {"course_name": "A"})
        repo.save("s2", {"course_name": "B"})
        repo.save("s1", {"course_name": "C"})

        results = repo.find_by_session("s1")
        assert len(results) == 2
        assert results[0]["course_name"] == "A"
        assert results[1]["course_name"] == "C"

    def test_find_by_session_empty(self) -> None:
        repo = InMemoryEnrollmentRepository()
        assert repo.find_by_session("nonexistent") == []

    def test_saved_enrollment_has_default_status_and_progress(self) -> None:
        repo = InMemoryEnrollmentRepository()
        repo.save("s1", {"course_name": "X", "seats": 2, "total_price": 50.0})
        results = repo.find_by_session("s1")
        assert results[0]["status"] == "confirmed"
        assert results[0]["progress"] == "not started"
        assert results[0]["starts_in"] == "next cohort, 2 weeks"

    def test_reset_clears_data_and_counter(self) -> None:
        repo = InMemoryEnrollmentRepository()
        repo.save("s1", {})
        repo.reset()
        assert repo.find_by_session("s1") == []
        assert repo.save("s1", {}) == "ENR-1000"


class TestEnrollmentService:
    def test_create_enrollment_delegates_to_repo(self) -> None:
        repo = InMemoryEnrollmentRepository()
        service = EnrollmentService(repo)
        enrollment_id = service.create_enrollment(
            "s1", {"course_name": "Test", "seats": 1, "total_price": 10.0}
        )
        assert enrollment_id.startswith("ENR-")

    def test_find_enrollments_delegates_to_repo(self) -> None:
        repo = InMemoryEnrollmentRepository()
        service = EnrollmentService(repo)
        service.create_enrollment("s1", {"course_name": "A"})
        service.create_enrollment("s1", {"course_name": "B"})
        assert len(service.find_enrollments("s1")) == 2
