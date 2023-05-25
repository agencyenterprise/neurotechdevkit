# Welcome to Neurotech Development Kit

The [_Neurotech Development Kit_ (NDK)](https://agencyenterprise.github.io/neurotechdevkit/) is an open-source software library designed to enhance accessibility to cutting-edge neurotechnology.
Featuring an easy-to-use API and pre-built examples, the NDK provides a seamless starting point for users.
Moreover, the NDK offers educational resources, such as interactive simulations and notebook-based tutorials, catering to a diverse audience including researchers, educators, engineers, and trainees.
By lowering the barrier of entry for newcomers and accelerating the progress of researchers, the NDK aims to be a versatile and invaluable tool for the neurotech community.

The initial set of target users for the NDK are ultrasound simulation trainees – individuals with backgrounds in technical or neuroscience-related fields who are learning to perform ultrasound simulations.
Our goal is to help users familiarize themselves with ultrasound simulation, understand the importance of input parameters, and streamline the process of running and visualizing simulations.
In the future, we plan to expand the NDK's features to incorporate additional functionality and modalities, catering to a broader range of users, including ultrasound researchers, product developers, machine learning engineers, and many more.

The initial release of NDK provides support for transcranial functional ultrasound stimulation, with a focus on providing comprehensive documentation, API flexibility, and visualizations.
The Neurotech Development Kit is actively developed and we welcome feedback and contributions.

![Simulation](https://raw.githubusercontent.com/agencyenterprise/neurotechdevkit/main/docs/images/ndk_example.gif)

Check out the [NDK documentation page](https://agencyenterprise.github.io/neurotechdevkit/).

## Running

### Docker

You can run `NDK` inside a docker container with a couple of steps:

1. Install [docker](https://docs.docker.com/engine/install/#desktop)

1. Execute `docker run -p 8888:8888 -v $(pwd)/notebooks:/ndk/notebooks -it ghcr.io/agencyenterprise/neurotechdevkit:latest`

  The command above will create a folder `notebooks` in your current directory where you can put your [Jupyter notebooks](https://docs.jupyter.org/en/latest/start/index.html) and start using `neurotechdevkit`.

  The output of the command above contains the URL of a jupyter notebook server, you can open the URL in your browser or connect to it using your IDE.

!!! note
    You can download a zip file containing notebook examples on this [link](https://agencyenterprise.github.io/neurotechdevkit/generated/gallery/gallery_jupyter.zip), and you can make them available into your container by extracting it into your local `notebooks` folder.

### Local installation

To install and run **neurotechdevkit** locally check the [installation](usage/installation.md) page.

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
