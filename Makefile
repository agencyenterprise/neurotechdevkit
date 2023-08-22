.PHONY:help lint lint-check test test-coverage test-unit test-integration docs web web-development

help:
	@echo "Available commands are: \n*lint, lint-check, spellcheck, test, test-unit, test-integration docs web web-development"

lint:
	poetry run isort web src tests docs/examples
	poetry run black web src tests docs/examples
	poetry run flake8 web src tests docs/examples
	poetry run mypy web src docs/examples
	poetry run codespell web src docs/examples
	poetry run pydocstyle web src
	poetry run pyright

spellcheck:
	poetry run pylint --disable all --enable spelling --spelling-dict en_US --spelling-private-dict-file=whitelist.txt src

lint-check:
	poetry run isort --check web src tests docs/examples
	poetry run black --check web src tests docs/examples
	poetry run flake8  web src tests docs/examples
	poetry run mypy web src docs/examples
	poetry run codespell web src docs/examples
	poetry run pydocstyle web src
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

web-development:
	poetry run flask --app web/app.py --debug run

web:
	poetry run gunicorn -b 0.0.0.0 -w 4 'web.app:app'