# Installation

You can run NDK without installing the package using docker, as shown [here](../index.md#running). However, if you'd like to install it, please follow the instructions below.

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


!!! note
    After installing `neurotechdevkit` you can use [Jupyter](https://jupyterlab.readthedocs.io/en/stable/) to play with the package.

    You can download notebook examples on this [link](https://agencyenterprise.github.io/neurotechdevkit/generated/gallery/gallery_jupyter.zip).

## MacOS

The two main compiler options for MacOS are **clang** and **gcc**.

### clang

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

### gcc

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
