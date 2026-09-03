.PHONY: build up down logs index inspect-documents inspect-embeddings test lint format clean eval eval-baseline

build:
	docker compose build

up:
	docker compose up

# Build the vector index — embeds the catalog into ChromaDB. Run once after `make up`
# (and again whenever data/courses.json changes). Starts the chroma service if needed.
index:
	docker compose run --rm course-agent python -m embedding.index

# Inspect what's stored in ChromaDB (read-only, no OpenAI key needed).
inspect-documents:
	docker compose run --rm course-agent python -m embedding.inspect --mode documents

inspect-embeddings:
	docker compose run --rm course-agent python -m embedding.inspect --mode embeddings

down:
	docker compose down

logs:
	docker compose logs -f

test:
	python -m pytest tests/ -v

lint:
	ruff check app/ embedding/ tests/

format:
	ruff format app/ embedding/ tests/

eval:
	python -m evaluation.skill_router_eval

# Re-run the eval against the pre-fix router prompt (reproduces the "before" number).
eval-baseline:
	python -m evaluation.skill_router_eval --baseline

clean:
	docker compose down -v
