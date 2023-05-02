# Notebook Philosophy


## Motivation

Jupyter notebooks are great for exploratory data analysis and experimentation. Often, when working in a notebook in a flow state, data scientists quickly move from cell to cell implementing and testing ideas. This is great for short-term learning, but the resulting notebook usually does not provide much context for a future reader to follow along and understand, and so the value of the notebook fades with time.

For a project that is long-lived or has a team which changes often, data scientists need to rely on knowledge built across many experiments and experimenters instead of individually learning everything themselves (as scientists must stand on the shoulders of giants). When building up a knowledge base from experiments in notebooks, the best way to ensure future value for the notebooks is to make each of them easily understandable by a future reader (which could be either a new team member, or the original author after months or years).


## Goals

The goals for our repository of notebooks are:

* each notebook contributes value to a growing knowledge base within the repository
* notebooks are easily interpretable by future readers
* notebooks provide enough detail and context so that any results in the notebook can be reproduced
* results and conclusions in notebooks are reliable as a foundation of knowledge for future notebooks
* previous results are not lost or overwritten


## Practices

We have found that the following practices help to achieve the above goals. 

* include a clear introduction to the notebook describing the objective and approach
  * include a link to the pivotal story
* include a conclusion to the notebook describing the most important learnings and next steps
* be consistent with notebook naming to make notebooks more easily findable
  * our current naming pattern is the pivotal id followed by a lower-case hyphen-separated description
* spread markdown text throughout the notebook to explain context and reasoning
* aim to make notebooks self-contained and independently understandable where reasonable
  * for example: if comparing to previous results, pull in the previous results rather than referencing the other notebook
  * but avoid duplicating previous work or re-writing large amounts of context, in these situations referencing with a link is preferred
* clean up and re-read notebooks from start to end before finalizing a PR
* make code readable and use functions to reduce duplication and minimize top-level code in cells
* aim to make the notebook re-runnable from start to finish with no errors and identical results, ideally even on another team member's machine
  * always explicitly set seeds wherever randomization is used
  * prefer the use of relative paths to consistent file locations within the repository tree
* treat notebooks as read-only once their PR is accepted
  * put additions and follow-ups in new notebooks, although small corrections can sometimes (rarely) be made to original notebooks
* use a clear 1-to-1 correspondence between notebooks and research stories
* aim to keep notebooks and stories short and focused on one thing
  * notebooks should add incremental value rather than comprehensive research
  * stories and notebooks should focus either on research or engineering tasks but not both at the same time
* be pragmatic about these practices and keep focus on the goals rather than spending a lot of extra time following practices to the T

*also take a look at the [research story PR requirements](contributing.md#research-stories).*
