# Welcome to Neurotech Development Kit

[![Linting](https://github.com/agencyenterprise/neurotechdevkit/actions/workflows/lint.yml/badge.svg)](https://github.com/agencyenterprise/neurotechdevkit/actions/workflows/lint.yml)
[![Tests](https://github.com/agencyenterprise/neurotechdevkit/actions/workflows/test.yml/badge.svg)](https://github.com/agencyenterprise/neurotechdevkit/actions/workflows/test.yml)
[![Gallery examples](https://circleci.com/gh/agencyenterprise/neurotechdevkit.png?style=shield)](https://circleci.com/gh/agencyenterprise/neurotechdevkit)

The [_Neurotech Development Kit_ (NDK)](https://agencyenterprise.github.io/neurotechdevkit/) is an open-source and community-driven software library designed to lower the barrier of entry to cutting-edge neurotechnology for all enthusiasts, from scientists to clinicians and software hackers.

This package enables those fascinated with the future of neurotechnology and humanity to explore next-generation technologies without needing access to a laboratory or expensive medical equipment. As a community-driven project, we encourage you to contribute code, feedback, and features to help develop the future of human-machine communication and accelerate us towards a better future.

To empower users of any technical level, NDK features an easy-to-use API and a range of ready-to-use examples. And for advanced neuro-developers, it provides precise control of low-level functionality and an easy-to-expand framework. NDK is a rapidly developing project with the aim of attracting budding researchers, engineers, academics, and hackers to work on neurotechnology. To this end, we plan to rapidly expand the functionality of the package, build a friendly and empowered community of contributors, help guide new contributors, run machine-learning competitions, host hackathons, and much more.

The initial release of NDK provides support for transcranial functional ultrasound stimulation, with a focus on providing comprehensive documentation, API flexibility, and visualizations like the one below.

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

!!! note
    You can have persisting [Jupyter notebooks](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html) by running
    ```
    docker run -p 8888:8888  -v $(pwd)/notebooks:/ndk/notebooks -it ghcr.io/agencyenterprise/neurotechdevkit:latest
    ```
    The command above will create a folder `notebooks` in your current directory where you can put your jupyter notebooks.

    We recommend downloading the `.zip` file with example notebooks from this [link](https://agencyenterprise.github.io/neurotechdevkit/generated/gallery/gallery_jupyter.zip), and extracting it into your local `notebooks` folder so you can access them from the docker.

### Local installation

To install and run **neurotechdevkit** locally check the [installation](https://agencyenterprise.github.io/neurotechdevkit/usage/installation/) page.

## Usage

```python
import neurotechdevkit as ndk

scenario = ndk.make('scenario-0-v0')
result = scenario.simulate_steady_state()
result.render_steady_state_amplitudes(show_material_outlines=False)
```

![Simulation](https://raw.githubusercontent.com/agencyenterprise/neurotechdevkit/main/docs/images/simulation_steady_state.png)

## Acknowledgements

Thanks to Fred Ehrsam for supporting this project, Quintin Frerichs and Milan Cvitkovic for providing direction, and to Sumner Norman for his ultrasound and neuroscience expertise. Thanks to [Stride](https://www.stride.codes/) for facilitating ultrasound simulations and providing an MIT license for usage within NDK, [Devito](https://www.devitoproject.org/) for providing the backend solver, [Napari](https://napari.org/stable/) for great 3D visualization, and to [Jean-Francois Aubry, et al.](https://doi.org/10.1121/10.0013426) for the basis of the simulation scenarios.
