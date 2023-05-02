# Contributing

In the current phase of NDK, we are only accepting contributions from invited contributors. We will create a public production `ndk` repository when we are ready to open it up for additional contributions. At that time, we will update these contribution guidelines as appropriate.

We use [Pivotal Tracker](https://www.pivotaltracker.com/n/projects/2589904) as our issue tracker. Once we open-source the project, we plan to switch to Github issues.

Please take a look at existing [conventions](conventions.md) used throughout the codebase.


## Pull Request Acceptance Requirements

As a research repository, we have two types of stories:
1. **research stories**: where the objective of the story is to gather information, learn something how to do something, or test hypotheses. These are usually completed inside jupyter notebooks, and are saved inside the `experiments` directory for future reference.
2. **engineering stories**: where the objective of the story is to produce working code. These usually include contributions to the `neurotechdevkit` package.

The two story types have different PR acceptance requirements.


### Research Stories

All of the following criteria should be met for any PRs with research notebooks and associated code. Also check out our [notebook philosophy](notebook-philosophy.md) for guidelines (not required, but encouraged).

1. the content of the notebook fulfills the objective of the story and meets the story *ac* (acceptance criteria)
1. the notebook is runnable from start to finish and all results are reproducible\*
1. code and discussion in the notebook is clean and clear enough for a team member with reasonable context to easily follow
1. a markdown header is included with a title, date, link to the pivotal story, and an introduction to the notebook
1. the notebook name is composed of the pivotal story id followed by a hyphen-separated description (eg. `123456789-this-is-a-research-story.ipynb`)
1. a markdown conclusion is included summarizing the results and listing next steps

*note: these acceptance criteria do not apply for [spike](https://en.wikipedia.org/wiki/Spike_%28software_development%29) notebooks*


\* in exceptional cases where making the notebook runnable from start to finish would take an extraordinary amount of extra time, it is acceptable to forgo the runnable requirement. In these cases the exception needs to be clearly explained in a markdown cell and the notebook must contain enough information to reproduce all results even if it takes a bit of manual input.


### Engineering Stories

We want to make it easy for outside users to eventually contribute code to NDK. Ensuring that NDK has clean code with good documentation and good unit tests goes a long way towards meeting that goal. Therefore, we have strict requirements on new code.

1. the contribution fulfills the objective of the story and meets the story *ac* (acceptance criteria)
1. all linting checks pass:
    ```
    $ make lint
    ```
    * current checks include `isort`, `black`, `flake8` and `mypy`.
1. all unit tests pass:
    ```
    $ make test
    ```
    * test coverage on new code is appropriate and test coverage on existing code does not unnecessarily decrease
    * we don't currently have specific metrics or criteria for what is "appropriate", but [here is a nice story](https://stackoverflow.com/questions/90002/what-is-a-reasonable-code-coverage-for-unit-tests-and-why/90021#90021)
    * unit tests are not currently required on code which build visualizations as they are often difficult to test in an automated fashion but easy to test visually. We might change this guideline in the future
1. all classes and functions have clear docstrings following google-style conventions
    * docstrings should follow [PEP257 Docstring Conventions](https://peps.python.org/pep-0257/) and the [Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings)
    * module docstrings are not currently required, but will likely be added in the future
1. type hinting is used on all function and method parameters and return values, excluding tests
    * type hinting in test fixtures is encouraged where it would greatly increase clarity of the code
1. code conforms to [PEP8](https://peps.python.org/pep-0008/) (unless it conflicts with any of the previous requirements)
1. Clean Code practices are followed:
   * [authoritative source](https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship-ebook-dp-B001GSTOAM/dp/B001GSTOAM)
   * [summary list](https://gist.github.com/wojteklu/73c6914cc446146b8b533c0988cf8d29)

*note: some existing code does not meet all of these requirements as the requirements have evolved throughout the project. No need to fix any unmodified code, but please do update any sections of code that you touch on in a PR.*


### Continuous Integration (CI)

We plan to implement automated CI checks for PRs in the near future. See [#184241486](https://www.pivotaltracker.com/story/show/184241486). All CI checks will be required to pass. The checks we plan to include are:

1. linting:
    1. [isort](https://github.com/PyCQA/isort) (in check-only mode)
    1. [black](https://github.com/psf/black) (in check-only mode)
    1. [flake8](https://github.com/PyCQA/flake8)
    1. [mypy](https://github.com/python/mypy)
    1. [pydocstyle](https://github.com/PyCQA/pydocstyle)
    1. [codespell](https://github.com/codespell-project/codespell)
1. testing:
    1. all tests are executed using [pytest](https://github.com/pytest-dev/pytest)
    1. executed tests include both unit and integration tests


## Pull Request Process


### When authoring PRs:

* use PRs for all research and engineering stories
* ensure the PR meets the [acceptance requirements](#pull-request-acceptance-requirements) above
* use one branch per story, ideally branching off `main` and always merging back into `main`
    * if a story is dependent on a previously story which has not yet been merged, it is ok to branch off the branch for the dependency
* include the pivotal ID in the PR title and a link to the pivotal story in the PR description
* let the reviewer know about any details you think they should be aware of or you'd like them to take a look at by adding it to the description
    * no need to copy any information from the pivotal story to the PR - the reviewer should look at the story
* copy the PR link to the CODE field on the pivotal story
* request reviews from reviewers so that they receive notification
* address any requests for changes and re-request reviews when it is ready for another review
* once all requested reviews accept the PR, the author (not the reviewer) merges the PR
    * sometimes reviewers will leave minor comments, but accept the PR as-is. If this is the case, the author can make small updates addressing the comment if they wish (but are not required to) and do not need to request a new review before merging
* merge PRs using "Squash and Merge" method in order to keep the git history simple and clean
* once the PR is merged, delete the branch, and deliver and accept the story on pivotal


### When reviewing PRs:

* put priority on reviewing other's PRs in order to complete and get things merged quickly
* if the story is a research story, review using ReviewNB (a link should be automatically posted to the PR)
    * Make sure that once you've submitted the ReviewNB review you also create a review on the Github PR that specifies `comment`/`approve`/`request changes` as appropriate so that the author is notified
* if a story is an engineering story, be sure to check out the branch on your instance or local machine and make sure all unit tests and lint checks pass (this will no longer be required once we have CI in place)
* ensure the PR meets the [acceptance requirements](#pull-request-acceptance-requirements) for the story
* be sure to leave comments for things that are done well in addition to things that should be improved
* signal to the author that your review is complete by either accepting the PR as-is or requesting changes


### Work-In-Progress (WIP) PRs:

* feel free to open WIP PRs whenever you like
* use a Draft PR on github and also put WIP in the PR title
* if you want someone to review a WIP, request a review from them on github and also over-communicate directly to the reviewer what feedback you are looking for
    * normally we don't review WIP work to avoid spending time commenting on something that would already be fixed in the final PR
    * over-communicating about the desired feedback helps the reviewer focus on the feedback that is useful to you
* when reviewing a WIP PR, feel free to `comment` and `request changes`, but save `approve` for finalized PRs
