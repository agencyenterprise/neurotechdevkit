# Contributing

You can contribute to NDK by creating GitHub issues or by submitting pull requests.

## Reporting issues

Feel free to open an issue if you would like to discuss a new feature request or report a bug. When creating a bug report, please include as much information as possible to help us reproduce the bug as well as what the actual and expected behavior is.

## Contributing code

### Standards

To ensure efficient collaborative development, a variety of standards are utilized in this project.

- [Black](https://github.com/psf/black) code formatter is used.
- [Flake8](https://flake8.pycqa.org) is used for linting.
- [isort](https://pycqa.github.io/isort/) is used for sorting the imports.
- [pyright](https://github.com/microsoft/pyright) is used for static type checking.
- [Type hinting](https://docs.python.org/3/library/typing.html) is used.
      - And checked using [mypy](http://mypy-lang.org).

### Preparing your environment

Start by cloning the repository:

```
git clone https://github.com/agencyenterprise/neurotechdevkit.git
cd neurotechdevkit
```

#### Running on docker

If you don't want to install NDK's dependencies on your machine, you can run it in a container:

* Install [Docker](https://docs.docker.com/engine/install/#desktop).

* Run the container, which will start a jupyter notebook server:
   ```
   docker compose up
   ```

* Connect to the jupyter notebook directly in your browser or with your IDE.

#### Running locally

This project requires Python `>=3.9` and `<3.11` to be installed. You can find the Python version you have installed by running `python --version` in a terminal. If you don't have Python installed or are running an unsupported version, you can download a supported version from [python.org](https://www.python.org/downloads/).

We use [poetry](https://python-poetry.org/) to manage dependencies and virtual environments. Follow the instructions from [poetry's documentation](https://python-poetry.org/docs/#installation) to install it if you don't have it on your system.

Install the dependencies by running the following command in a shell within the project directory:

```
poetry install
```

This will resolve and install the dependencies from `poetry.lock` and will install the `neurotechdevkit` package in editable mode.


Install stride with

```bash
$ poetry run pip install git+https://github.com/trustimaging/stride
```

### Using the environment

If you are not already using a virtual environment, `poetry` will [create one for you by default](https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment). You will need to use this virtual env when using or working on the package.

Activate the environment directly via:

```
poetry shell
```

If you are already using your own virtual environment, you should not need to change anything.

## Code requirements and conventions

!!! note
      The following commands require `GNU make` to be installed, on Windows you can install it with [Chocolatey](https://chocolatey.org/install):

      `choco install make`

Before opening a pull request, please make sure that all of the following requirements are met:

1. all unit and integration tests are passing:
   ```
   make test
   ```
1. the code is linted and formatted:
   ```
   make lint
   ```
1. spelling is checked:
   ```
   make spellcheck
   ```
1. the documentation builds without warnings:
   ```
   make docs
   ```
1. type hinting is used on all function and method parameters and return values, excluding tests
1. docstring usage conforms to the following:
      1. all docstrings should follow [PEP257 Docstring Conventions](https://peps.python.org/pep-0257/)
      1. all public API classes, functions, methods, and properties have docstrings and follow the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
      1. docstrings on private objects are not required, but are encouraged where they would significantly aid understanding
1. testing is done using the pytest library, and test coverage should not unnecessarily decrease.
