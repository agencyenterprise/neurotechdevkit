# -*- coding: utf-8 -*-
"""
Save and load results
====================================

This example demonstrates how a simulation can be executed and stored in one
computer and the results can be loaded and rendered in another computer.
"""
# %%
# Execute the following code in a computer with ndk installed
import neurotechdevkit as ndk

scenario = ndk.make('scenario-0-v0')
result = scenario.simulate_steady_state()
result.save_to_disk('scenario-0-v0-results.tar.gz')


# %%
# The output file from the previous step could be copied to
# another computer with ndk installed or just stored for future use. The results can
# be loaded running the following code:
import neurotechdevkit as ndk

result = ndk.load_result_from_disk('scenario-0-v0-results.tar.gz')
result.render_steady_state_amplitudes(show_material_outlines=False)
# %%
