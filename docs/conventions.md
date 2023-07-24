# Conventions

Below are some of the conventions that we are or would like to follow in NDK. This page should be considered incomplete as there are conventions being followed in the codebase which are not described here. We should add to this doc incrementally as part of relevant PRs.


## Units of Measurement

Unit of measurement (uom) labels should be used wherever relevant throughout docstrings, documentation, plots, and textual output.

For internal methods and computation steps, always express values using uom without scale prefixes (eg. m rather than mm or km). When values are provided by the user (via parameters) or shown directly to the user (via plots or textual output) scaling prefixes can be used if there is a clear convention in the ultrasound community for which prefix should be used.


### In docstrings

* include units for all parameters and return values that have meaningful units
    * put the uom in parentheses after "in". Eg: "distance (in meters)"
* when the uom has an SI name, write out the name fully with the correct capitalization (eg. seconds, meters, pascals)
* when the uom does not have an SI name, use the equation specifying the uom (eg. m/s, W/cm²)


### In plots

* for axis labels, use the uom abbreviation in parentheses. eg: "Pressure (Pa)"


### For metrics

* include a specific uom in the output for each metric


## Adding to Conventions

Whenever new conventions are used, they should be added to this document within the PR where the conventions were first added.

Whenever existing conventions are discovered or refined (such as through PR review discussion), the conventions should be added or updated in this document as part of that PR.

