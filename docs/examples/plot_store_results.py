# -*- coding: utf-8 -*-
"""
Save and load results
====================================

!!! note
    NDK and its examples are under constant development, more information and
    content will be added to this example soon!

This example demonstrates how a simulation can be executed and stored in one
computer and the results can be loaded and rendered later in the same computer
or another one.
"""
# %%
# Execute the following code in a computer with ndk installed
import neurotechdevkit as ndk

scenario = ndk.built_in.ScenarioSimple()
scenario.make_grid()
scenario.compile_problem()
result = scenario.simulate_steady_state()
result.save_to_disk("scenario-0-results.tar.gz")


# %%
# The output file from the previous step could be copied to
# another computer with ndk installed or stored for future use. The results can
# be loaded running the following code:
import neurotechdevkit as ndk  # noqa: E402

result2 = ndk.load_result_from_disk("scenario-0-results.tar.gz")
assert isinstance(result2, ndk.results.SteadyStateResult2D)
result2.render_steady_state_amplitudes(show_material_outlines=False)
# %%
