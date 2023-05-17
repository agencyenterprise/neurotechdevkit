.PHONY:help lint lint-check test test-coverage test-unit test-integration docs

help:
	@echo "Available commands are: \n*lint, lint-check, test, test-unit, test-integration docs"

lint:
	poetry run isort src tests
	poetry run black src tests
	poetry run flake8 src tests
	poetry run mypy src
	poetry run codespell src
	poetry run pydocstyle src
	poetry run pyright

lint-check:
	poetry run isort --check  src tests
	poetry run black --check  src tests
	poetry run flake8  src tests
	poetry run mypy src
	poetry run codespell src
	poetry run pydocstyle src
	poetry run pyright --warnings

test:
	poetry run pytest tests

test-coverage:
	poetry run pytest . --color=yes --ignore=experiments --ignore=BKP -m "not jitter" --cov=src/neural_data_simulator --cov-report=term-missing:skip-covered --junitxml=pytest.xml --cov-report=xml 2>&1 | tee pytest-coverage.txt

test-unit:
	poetry run pytest tests -m "not integration"

test-integration:
	poetry run pytest tests -m "integration"

docs:
	poetry run mkdocs build