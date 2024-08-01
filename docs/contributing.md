# Contributing

You can contribute to NDK by creating [GitHub issues](https://github.com/agencyenterprise/neurotechdevkit/issues) or by submitting [pull requests](https://github.com/agencyenterprise/neurotechdevkit/pulls).

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
- [Pylint](https://pypi.org/project/pylint/) is used for spell checking.

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
   docker compose up ndk
   ```

* Connect to the jupyter notebook directly in your browser or with your IDE.


Alternatively, you can start the web user interface with:

   ```
   docker compose up ndk-ui
   ```

* Open the address `http://127.0.0.1:8080/` in your browser to access it.


#### Running locally

This project requires Python `>=3.9` and `<3.11` to be installed. You can find the Python version you have installed by running `python --version` in a terminal. If you don't have Python installed or are running an unsupported version, you can download a supported version from [python.org](https://www.python.org/downloads/).

We use [poetry](https://python-poetry.org/) to manage dependencies and virtual environments. Follow the instructions from [poetry's documentation](https://python-poetry.org/docs/#installation) to install it if you don't have it on your system.

Install the dependencies by running the following command in a shell within the project directory:

```
poetry install
```

This will resolve and install the dependencies from `poetry.lock` and will install the `neurotechdevkit` package in editable mode.


Install stride with:

```bash
$ poetry run pip install git+https://github.com/trustimaging/stride@2520c59
```

Follow the steps described in [Setting up a compiler](installation.md#setting-up-a-compiler).

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
1. type hinting is used on all function and method parameters and return values, excluding tests
1. docstring usage conforms to the following:
      1. all docstrings should follow [PEP257 Docstring Conventions](https://peps.python.org/pep-0257/)
      2. all public API classes, functions, methods, and properties have docstrings and follow the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
      3. docstrings on private objects are not required, but are encouraged where they would significantly aid understanding
1. testing is done using the pytest library, and test coverage should not unnecessarily decrease.


## Process

### Versioning

NDK uses [semantic versioning](https://en.wikipedia.org/wiki/Software_versioning#Semantic_versioning) to identify its releases.

We use the [release on push](https://github.com/rymndhng/release-on-push-action/tree/master/) github action to generate the new version for each release. This github action generates the version based on a pull request label assigned before merge. The supported labels are:

- `release-patch`
- `release-minor`
- `release-major`
- `norelease`

### Automatic release

Merged pull requests with one of the labels `release-patch`, `release-minor` or `release-major` will trigger a release job on CI.

The release job will:

1. generate a new package version using semantic versioning provided by [release on push](https://github.com/rymndhng/release-on-push-action/tree/master/)
1. update the `pyproject.toml` version using `poetry`
1. commit the updated `pyproject.toml` file using the [git-auto-commit action](https://github.com/stefanzweifel/git-auto-commit-action/tree/v4/)
1. push the package to pypi using [poetry publish](https://github.com/JRubics/poetry-publish)
1. build a new docker image and tag it with the previously generated semantic version

Pull requests merged with the tag `norelease` will not trigger any of the actions listed above.

### Gallery of examples

The examples you can find in the official [documentation](http://ndk-docs.s3-website.us-east-2.amazonaws.com/generated/gallery/) are [python scripts](https://github.com/agencyenterprise/neurotechdevkit/tree/main/docs/examples) executed in CI.

Running these scripts is a resource intensive and time consuming task, for this reason we are using CircleCI instead of Github Actions (as we can choose a more powerful machine to execute the job).

### Checking NDK documentation on CI

All pull requests trigger a CI job that builds the documentation and makes the built files available.

To check the generated documentation in a pull request:

1. Scroll to the bottom of the page and click on the `Show all checks` link.
1. Click on the details link of the `Check the rendered docs here!` job.
      <figure markdown>
            ![circle-ci-rendered-docs](images/circle-ci-rendered-docs.png){ width="800" }
      </figure>

!!! note
      The `Examples` section is not properly rendered when the documentation is built
      on CI. The links of the thumbnails in `gallery/index.html` point to broken paths,
      in order to check one of the examples you will have to click on the left panel,
      as shown in the image below:
      <figure markdown>
            ![gallery-link](images/gallery-link.png){ width="500" }
      </figure>
      Within each example, the outputs of cells are also not properly displayed.
