# Welcome to Neurotech Development Kit

The _Neurotech Development Kit_ (NDK) is an open-source software library designed to enhance accessibility to cutting-edge neurotechnology.
Featuring an easy-to-use API and pre-built examples, the NDK provides a seamless starting point for users.
Moreover, the NDK offers educational resources, such as interactive simulations and notebook-based tutorials, catering to a diverse audience including researchers, educators, engineers, and trainees.
By lowering the barrier of entry for newcomers and accelerating the progress of researchers, the NDK aims to be a versatile and invaluable tool for the neurotech community.

The initial set of target users for the NDK are ultrasound simulation trainees – individuals with backgrounds in technical or neuroscience-related fields who are learning to perform ultrasound simulations.
Our goal is to help users familiarize themselves with ultrasound simulation, understand the importance of input parameters, and streamline the process of running and visualizing simulations.
In the future, we plan to expand the NDK's features to incorporate additional functionality and modalities, catering to a broader range of users, including ultrasound researchers, product developers, machine learning engineers, and many more.

The initial release of NDK provides support for transcranial functional ultrasound stimulation, with a focus on providing comprehensive documentation, API flexibility, and visualizations.
The Neurotech Development Kit is actively developed and we welcome feedback and contributions.

![Simulation](https://raw.githubusercontent.com/agencyenterprise/neurotechdevkit/main/docs/images/ndk_example.gif)

## Running

### Local installation

`neurotechdevkit` requires Python `>=3.9` and `<3.11` to be installed. You can find which Python version you have installed by running `python --version` in a terminal.

If you don't have Python installed, or you are running an unsupported version, you can download it from [python.org](https://www.python.org/downloads/). Python environment managers like pyenv, conda, and poetry are all perfectly suitable as well.

You can install the `neurotechdevkit` package using:

```bash
pip install neurotechdevkit
```

You also have to install stride, it can be done running:

```bash
pip install git+https://github.com/trustimaging/stride
```

`devito`, a dependency of `neurotechdevkit`, requires `libomp`. On MacOS it can be installed with:

```
brew install libomp
```

the output of the command above will look like this:

```
For compilers to find libomp you may need to set:
export LDFLAGS="-L/usr/local/opt/libomp/lib"
export CPPFLAGS="-I/usr/local/opt/libomp/include"
```

`devito` requires the directory with `libomp` headers to be accessible during the runtime compilation, you can make it accessible by exporting a new environment variable `CPATH` with the path for libomp headers, like so:

```
export CPATH="/usr/local/opt/libomp/include"
```

You will also have to set an environment variable that defines what compiler `devito` will use, like so:

```
export DEVITO_ARCH=gcc
```

the supported values for `DEVITO_ARCH` are: `'custom', 'gnu', 'gcc', 'clang', 'aomp', 'pgcc', 'pgi', 'nvc', 'nvc++', 'nvidia', 'cuda', 'osx', 'intel', 'icpc', 'icc', 'intel-knl', 'knl', 'dpcpp', 'gcc-4.9', 'gcc-5', 'gcc-6', 'gcc-7', 'gcc-8', 'gcc-9', 'gcc-10', 'gcc-11'`

### Docker

You can run `NDK` inside a docker container with a couple of steps:

1. Install [docker](https://docs.docker.com/engine/install/#desktop)

1. Execute `docker run -p 8888:8888 ghcr.io/agencyenterprise/neurotechdevkit:latest`

The output of the command above contains the URL of a jupyter notebook server, you can open the URL in your browser or connect to it using your IDE.


## Usage

```python
import neurotechdevkit as ndk

scenario = ndk.make('scenario-0-v0')
result = scenario.simulate_steady_state()
result.render_steady_state_amplitudes(show_material_outlines=False)
```

![Simulation](https://raw.githubusercontent.com/agencyenterprise/neurotechdevkit/main/docs/images/simulation_steady_state.png)

### Troubleshooting

#### `Error: Process completed with exit code 1.` when installing Stride on Windows

  Unfortunately Stride can't be installed on a Windows platform, therefore NDK is also unsupported.

#### Getting error `codepy.CompileError: module compilation failed`

  This error occurs when the compiler wasn't able to perform the compilation, it can be caused by a environment configuration problem. Check the `DEVITO_ARCH` environment variable, it should be set with the compiler devito will use to compile the code.

  You can find further information in the [Devito](https://github.com/devitocodes/devito/wiki/) documentation.

#### Getting error `ModuleNotFoundError: No module named 'neurotechdevkit'`

  This error is shown when `neurotechdevkit` is not installed, if you installed it using a virtual environment like poetry you must run the script with `poetry run` or activate the environment.

#### Getting error `AttributeError: module 'napari' has no attribute 'Viewer'` when calling `render_layout_3d`

  This error is shown when napari is not installed, make sure to run

  `poetry run pip install "napari[all]"`

  and try again.

### Acknowledgements

Thanks to Fred Ehrsam for supporting this project, Quintin Frerichs and Milan Cvitkovic for providing direction, and to Sumner Norman for his ultrasound and neuroscience expertise. Thanks to [Stride](https://www.stride.codes/) for facilitating ultrasound simulations, [Devito](https://www.devitoproject.org/) for providing the backend solver, [Napari](https://napari.org/stable/) for great 3D visualization, and to [Jean-Francois Aubry, et al.](https://doi.org/10.1121/10.0013426) for the basis of the simulation scenarios.
