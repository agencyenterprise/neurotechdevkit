In order to use your GPU to run NDK simulations you will have to install the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk-downloads).

!!! warning
    Make sure the [HPC SDK environment variables](https://docs.nvidia.com/hpc-sdk/hpc-sdk-install-guide/index.html#install-linux-end-usr-env-settings) are exported.

You will only have to export one environment variable to enable the GPU support for NDK:

```bash
export PLATFORM="nvidia-acc"
```

Now when running NDK simulations you should be able to see `platform=nvidiaX` in the execution output:

```py
import neurotechdevkit as ndk

scenario = ndk.scenarios.built_in.ScenarioRealisticSkull_2D()
scenario.make_grid()
scenario.compile_problem()
result = scenario.simulate_steady_state()
result.render_steady_state_amplitudes()
```

Output:
```bash
...

Operator `acoustic_iso_state` instance configuration:
	 * subs={h_x: 0.0005, h_y: 0.0005}
	 * opt=advanced
	 * compiler=pgcc
	 * language=openacc
	 * platform=nvidiaX

...
```

!!! warning
	Simulations with high memory requirement (e.g. 3D simulations) may not fit in the GPU and running them with GPU acceleration might crash the simulation.
