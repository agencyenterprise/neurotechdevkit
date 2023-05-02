The Neurotech Development Kit provides support for a range of simulation modes, including pulsed simulation and steady-state simulation.

Running a simulation takes just a single function call.

```py
import neurotechdevkit as ndk

scenario = ndk.make('scenario-2-2d-v0')
result = scenario.simulate_steady_state()
result.render_steady_state_amplitudes()
```

<figure markdown>
  ![Scenario Layout](../images/steady_state_wave_amplitude.png){ width="600" }
</figure>
