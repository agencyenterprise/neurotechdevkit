.PHONY:help lint lint-check test test-unit test-integration docs

help:
	@echo "Available commands are: \n*lint, lint-check, test, test-unit, test-integration docs"

lint:
	poetry run isort src tests
	poetry run black src tests
	poetry run flake8 src tests
	poetry run mypy src
	poetry run codespell src
	poetry run pydocstyle src

lint-check:
	poetry run isort --check  src tests
	poetry run black --check  src tests
	poetry run flake8  src tests
	poetry run mypy src
	poetry run codespell src
	poetry run pydocstyle src

test:
	poetry run pytest tests

test-unit:
	poetry run pytest tests -m "not integration"

test-integration:
	poetry run pytest tests -m "integration"

docs:
	poetry run mkdocs build