# Welcome to Neurotech Development Kit

[![Linting](https://github.com/agencyenterprise/neurotechdevkit/actions/workflows/lint.yml/badge.svg)](https://github.com/agencyenterprise/neurotechdevkit/actions/workflows/lint.yml)
[![Tests](https://github.com/agencyenterprise/neurotechdevkit/actions/workflows/test.yml/badge.svg)](https://github.com/agencyenterprise/neurotechdevkit/actions/workflows/test.yml)
[![Gallery examples](https://circleci.com/gh/agencyenterprise/neurotechdevkit.png?style=shield)](https://circleci.com/gh/agencyenterprise/neurotechdevkit)

The [_Neurotech Development Kit_ (NDK)](https://agencyenterprise.github.io/neurotechdevkit/) is an [open-source](https://github.com/agencyenterprise/neurotechdevkit), community-driven software library designed to lower the barrier of entry to the next generation of neurotechnology for current researchers and companies. It also enables software developers without access to hardware and human subjects to solve open problems in the field. The initial release of NDK provides support for transcranial focused ultrasound stimulation, along with comprehensive documentation, API flexibility, and 2D/3D visualizations. Future areas of interest may include photoacoustic and optical whole-brain imaging.

As a community-driven project, we encourage you to contribute code, feedback, and features to help accelerate the development of transformative neurotechnology.

![Simulation](https://raw.githubusercontent.com/agencyenterprise/neurotechdevkit/main/docs/images/ndk_example.gif)

Check out the [NDK documentation page](https://agencyenterprise.github.io/neurotechdevkit/).

## Running

### Docker

You can run `neurotechdevkit` inside a docker container with just a couple of steps:

1. Install [docker](https://docs.docker.com/engine/install/#desktop)

1. Run the following command:

    ```
    docker run -p 8888:8888 -it ghcr.io/agencyenterprise/neurotechdevkit:latest
    ```

    The command above will start a [Jupyter notebook](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html) server with example notebooks you can use to explore `neurotechdevkit`. Use the printed URL to open it in your browser or connect to it using your IDE.

    All changes you make to these files will be lost once you stop the docker container.

> **Note**:
>
>    You can have persisting [Jupyter notebooks](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html) by running
>    ```
>    docker run -p 8888:8888  -v $(pwd)/notebooks:/ndk/notebooks -it ghcr.io/agencyenterprise/neurotechdevkit:latest
>    ```
>    The command above will create a folder `notebooks` in your current directory where you can put your jupyter notebooks.
>
>    We recommend downloading the `.zip` file with example notebooks from this [link](https://agencyenterprise.github.io/neurotechdevkit/generated/gallery/gallery_jupyter.zip), and extracting it into your local `notebooks` folder so you can access them from the docker.

### Local installation

To install and run **neurotechdevkit** locally check the [installation](https://agencyenterprise.github.io/neurotechdevkit/installation/) page.

## Usage

```python
import neurotechdevkit as ndk

scenario = ndk.scenarios.built_in.Scenario0()
scenario.make_grid()
scenario.compile_problem()
result = scenario.simulate_steady_state()
result.render_steady_state_amplitudes(show_material_outlines=False)
```

![Simulation](https://raw.githubusercontent.com/agencyenterprise/neurotechdevkit/main/docs/images/simulation_steady_state.png)

## Acknowledgements

Thanks to Fred Ehrsam for supporting this project, Quintin Frerichs and Milan Cvitkovic for providing direction, and to Sumner Norman for his ultrasound and neuroscience expertise. Thanks to [Stride](https://www.stride.codes/) for facilitating ultrasound simulations and providing an MIT license for usage within NDK, [Devito](https://www.devitoproject.org/) for providing the backend solver, [Napari](https://napari.org/stable/) for great 3D visualization, and to [Jean-Francois Aubry, et al.](https://doi.org/10.1121/10.0013426) for the basis of the simulation scenarios.
