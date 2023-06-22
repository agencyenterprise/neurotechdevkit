# Installation

You can run NDK without installing the package using docker, as shown [here](index.md#running). However, if you'd like to install it, please follow the instructions below.

??? "Before installing on Windows"

    1. Install [Ubuntu on WSL](https://ubuntu.com/tutorials/install-ubuntu-on-wsl2-on-windows-11-with-gui-support#1-overview).

    1. Follow the the `Linux` steps described in this page inside your Ubuntu shell.

`neurotechdevkit` requires Python `>=3.9` and `<3.11` to be installed. You can find which Python version you have installed by running `python --version` in a terminal.

If you don't have Python installed, or you are running an unsupported version, you can download it from [python.org](https://www.python.org/downloads/). Python environment managers like pyenv, conda, and poetry are all perfectly suitable as well.

??? "Before installing on Linux"

    1. In order to install `neurotechdevkit` you must first install `g++` and the `python-dev` package for your python version.

        Both packages can be installed with:
        ```
        apt-get install -y g++ python3.10-dev
        ```

        **Important:** You must replace `3.10` with your python version when running the command above.


You can install the `neurotechdevkit` package using:

```bash
pip install neurotechdevkit
```

You also have to install stride, it can be done running:

```bash
pip install git+https://github.com/trustimaging/stride
```

## Setting up a compiler

NDK uses [devito](https://www.devitoproject.org/devito/) to perform the heavy computational operations. Devito generates, compiles and runs C code to achieve better performance.
The compiler used by Devito has to be selected, and paths for the linker might also be added as environment variables.

As a last step **before running NDK**, follow the instructions below depending on your OS.

??? "Before running on MacOS"

    The two main compiler options for MacOS are **clang** and **gcc**.

    ### clang

    If you prefer to use **clang** you will have to install `libomp` and `llvm`, you will also have to export a few environment variables needed by the compiler.

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


    1. Run the following command to export a new environment variable `CPATH` with the path for `libomp` headers:

        ```
        echo 'export CPATH="/usr/local/opt/libomp/include"' >> ~/.zshrc
        ```

    1. Install `llvm`:

        ```
        brew install llvm
        ```

    1. Run the following commands to export the `llvm` environment variables:

        ```
        echo 'export PATH="/usr/local/opt/llvm/bin:$PATH"' >> ~/.zshrc
        echo 'export LDFLAGS="-L/usr/local/opt/llvm/lib"' >> ~/.zshrc
        echo 'export CPPFLAGS="-I/usr/local/opt/llvm/include"' >> ~/.zshrc
        ```

    1. The following command will export the `DEVITO_ARCH` environment variable:

        ```
        echo 'export DEVITO_ARCH="clang"' >> ~/.zshrc
        ```

    1. Load the modified zsh configuration file:

        ```
        source ~/.zshrc
        ```

    ### gcc

    On MacOS the `gcc` executable is a symbolic link to `clang`, so by defining ~~DEVITO_ARCH=gcc~~ devito will try to add `gcc` flags to the `clang` compiler, and the compilation will most probably fail.

    You can tell devito to use the correct `gcc` compiler doing the following:

    1. Install gcc-11

        ```
        brew install gcc@11
        ```

    1. Run the command that exports the `DEVITO_ARCH` environment variable:

        ```
        echo 'export DEVITO_ARCH="gcc-11"' >> ~/.zshrc
        ```

    1. Load the modified zsh configuration file:

        ```
        source ~/.zshrc
        ```


??? "Before running on Linux"

    1. Export the `DEVITO_ARCH` environment variable, or add it to your shell profile:

        ```
        export DEVITO_ARCH="gcc"
        ```

        The supported values for `DEVITO_ARCH` are: `'custom', 'gnu', 'gcc', 'clang', 'aomp', 'pgcc', 'pgi', 'nvc', 'nvc++', 'nvidia', 'cuda', 'osx', 'intel', 'icpc', 'icc', 'intel-knl', 'knl', 'dpcpp', 'gcc-4.9', 'gcc-5', 'gcc-6', 'gcc-7', 'gcc-8', 'gcc-9', 'gcc-10', 'gcc-11'`.


!!! note
    After installing `neurotechdevkit` you can use [Jupyter](https://jupyterlab.readthedocs.io/en/stable/) to explore the package.

    To get started, we recommend downloading the example notebooks from this [link](https://agencyenterprise.github.io/neurotechdevkit/generated/gallery/gallery_jupyter.zip).

    **On Linux** you can download and extract the notebooks running the following commands:

    1. `sudo apt-get update && sudo apt-get install -y unzip wget`
    1. `wget "https://agencyenterprise.github.io/neurotechdevkit/generated/gallery/gallery_jupyter.zip" -O temp.zip && unzip temp.zip && rm temp.zip`