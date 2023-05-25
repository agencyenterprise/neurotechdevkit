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


NDK uses [devito](https://www.devitoproject.org/devito/) to perform the heavy computational operations. Devito generates, compiles and runs C code to achieve better performance.
The compiler used by Devito has to be selected, and paths for the linker might also be added as environment variables.

Export the environment variable that defines what compiler `devito` will use:

```
export DEVITO_ARCH=gcc
```

The supported values for `DEVITO_ARCH` are: `'custom', 'gnu', 'gcc', 'clang', 'aomp', 'pgcc', 'pgi', 'nvc', 'nvc++', 'nvidia', 'cuda', 'osx', 'intel', 'icpc', 'icc', 'intel-knl', 'knl', 'dpcpp', 'gcc-4.9', 'gcc-5', 'gcc-6', 'gcc-7', 'gcc-8', 'gcc-9', 'gcc-10', 'gcc-11'`


#### MacOS

The two main compiler options for MacOS are **clang** and **gcc**.

##### clang

If you prefer to use **clang** you will have to install `libomp` and `llvm`, you will also have to export a few environment variables needed for the compiler.

1. Install libomp

    ```
    brew install libomp
    ```

    the output of the command above will look like this:

    ```
    For compilers to find libomp you may need to set:
    export LDFLAGS="-L/usr/local/opt/libomp/lib"
    export CPPFLAGS="-I/usr/local/opt/libomp/include"
    ```


1. Export a new environment variable `CPATH` with the path for `libomp` headers, like so:

    ```
    export CPATH="/usr/local/opt/libomp/include"
    ```

1. Install `llvm`:

    ```
    brew install llvm
    ```

1. Export the following environment variables:

    ```
    export PATH="/usr/local/opt/llvm/bin:$PATH"
    export LDFLAGS="-L/usr/local/opt/llvm/lib"
    export CPPFLAGS="-I/usr/local/opt/llvm/include"
    ```

1. Export the `DEVITO_ARCH` environment variable

    ```
    export DEVITO_ARCH="clang"
    ```

##### gcc

On MacOS the `gcc` executable is a symbolic link to `clang`, so by defining ~~DEVITO_ARCH=gcc~~ devito will try to add `gcc` flags to the `clang` compiler, and the compilation will most probably fail.

You can tell devito to use the correct gcc compiler doing the following:

1. Install gcc-11

    ```
    brew install gcc@11
    ```

1. Export the `DEVITO_ARCH` environment variable

    ```
    export DEVITO_ARCH="gcc-11"
    ```

!!! note
    After installing `neurotechdevkit` you can use [Jupyter](https://docs.jupyter.org/en/latest/start/index.html) to play with the package.

    You can download notebook examples on this [link](https://agencyenterprise.github.io/neurotechdevkit/generated/gallery/gallery_jupyter.zip).


### Docker

You can run `NDK` inside a docker container with a couple of steps:

1. Install [docker](https://docs.docker.com/engine/install/#desktop)

1. Execute `docker run -p 8888:8888 -v $(pwd)/notebooks:/ndk/notebooks -it ghcr.io/agencyenterprise/neurotechdevkit:latest`

  The command above will create a folder `notebooks` in your current directory where you can put your [Jupyter notebooks](https://docs.jupyter.org/en/latest/start/index.html) and start using `neurotechdevkit`.

  The output of the command above contains the URL of a jupyter notebook server, you can open the URL in your browser or connect to it using your IDE.

!!! note
    You can download a zip file containing notebook examples on this [link](https://agencyenterprise.github.io/neurotechdevkit/generated/gallery/gallery_jupyter.zip), and you can make them available into your container by extracting it into your local `notebooks` folder.


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
