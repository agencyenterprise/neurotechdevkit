.PHONY:help lint lint-check test test-coverage test-unit test-integration docs web

help:
	@echo "Available commands are: \n*lint, lint-check, spellcheck, test, test-unit, test-integration docs web"

lint:
	poetry run isort src tests docs/examples
	poetry run black src tests docs/examples
	poetry run flake8 --exclude=src/web/app/ src tests docs/examples
	poetry run mypy --exclude=src/web/app/ src docs/examples
	poetry run codespell --skip="src/web/app" src docs/examples
	poetry run pydocstyle --match-dir="^(src/web/app)" src
	poetry run pyright

spellcheck:
	# If you are using an Apple Silicon Mac, there is currently an issue with
	# pyenchant (https://github.com/pyenchant/pyenchant/issues/265) that prevents
	# this spellcheck command from finding the enchant library (and en_US). 
	# Until pyenchant is fixed, one workaround is to run:
	#    `export PYENCHANT_LIBRARY_PATH=/opt/homebrew/lib/libenchant-2.dylib`
	# in the terminal environment before you run `make spellcheck`
	# See: https://github.com/pyenchant/pyenchant/issues/265#issuecomment-998965819
	poetry run pylint --disable all --enable spelling --spelling-dict en_US --spelling-private-dict-file=whitelist.txt src

lint-check:
	poetry run isort --check src tests docs/examples
	poetry run black --check src tests docs/examples
	poetry run flake8 --exclude=src/web/app/  src tests docs/examples
	poetry run mypy --exclude=src/web/app/ src docs/examples
	poetry run codespell --skip="src/web/app" src docs/examples
	poetry run pydocstyle --match-dir="^(src/web/app)" src
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

web:
	poetry run ndk-ui
