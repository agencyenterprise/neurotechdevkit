# Neurotech Development Kit (NDK)

The _Neurotech Development Kit_ (NDK) is an open-source software library designed to enhance accessibility to cutting-edge neurotechnology.
Featuring an easy-to-use API and pre-built examples, the NDK provides a seamless starting point for users.
Moreover, the NDK offers educational resources, such as interactive simulations and notebook-based tutorials, catering to a diverse audience including researchers, educators, engineers, and trainees.
By lowering the barrier of entry for newcomers and accelerating the progress of researchers, the NDK aims to be a versatile and invaluable tool for the neurotech community.

The initial set of target users for the NDK are ultrasound simulation trainees – individuals with backgrounds in technical or neuroscience-related fields who are learning to perform ultrasound simulations.
Our goal is to help users familiarize themselves with ultrasound simulation, understand the importance of input parameters, and streamline the process of running and visualizing simulations.
In the future, we plan to expand the NDK's features to incorporate additional functionality and modalities, catering to a broader range of users, including ultrasound researchers, product developers, machine learning engineers, and many more.

The initial release of NDK provides support for transcranial functional ultrasound stimulation, with a focus on providing comprehensive documentation, API flexibility, and visualizations.
The Neurotech Development Kit is actively developed and we welcome feedback and contributions.


## Exploring the Repository

The NDK API was developed with easy-of-use in mind, and **3 lines of code is all you need to run a simulation**:

```python
import neurotechdevkit as ndk

scenario = ndk.make('scenario-2-2d-v0')
result = scenario.simulate_steady_state()
```

... and one line to see the results!

```python
result.render_steady_state_amplitudes()
```

<img width="600" alt="steady-state-results" src="https://user-images.githubusercontent.com/90583560/227414328-4c529593-2d12-44f4-80d4-6a9c3a503d41.png">

## Setup

`neurotechdevkit` requires Python `>=3.9` and `<3.11` to be installed. You can find which Python version you have installed by running `python --version` in a terminal. If you don't have Python installed, or you are running an unsupported version, you can download it from [python.org](https://www.python.org/downloads/). Python environment managers like pyenv, conda, and poetry are all perfectly suitable as well.

You can install the package using:

``` bash
pip install neurotechdevkit
```

And then you must install stride using:
``` bash
pip install git+https://github.com/trustimaging/stride
```

### Development

See our [contribution requirements](docs/contributing.md) for more information on how to install the package locally using poetry and on how to contribute.

A Makefile is provided to assist with common commands such as linting and running unit tests.

```bash
$ make lint

$ make test
```

See the Makefile for other commands.

### Acknowledgements

Thanks to Fred Ehrsam for supporting this project, Quintin Frerichs and Milan Cvitkovic for providing direction, and to Sumner Norman for his ultrasound and neuroscience expertise. Thanks to [Stride](https://www.stride.codes/) for facilitating ultrasound simulations, [Devito](https://www.devitoproject.org/) for providing the backend solver, [Napari](https://napari.org/stable/) for great 3D visualization, and to [Jean-Francois Aubry, et al.](https://doi.org/10.1121/10.0013426) for the basis of the simulation scenarios.
