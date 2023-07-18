# Troubleshooting

This page contains a list of known problems you might face when installing and running NDK and the actions to solve them.

## `error: command 'x86_64-linux-gnu-gcc' failed: No such file or directory` on Linux

This error occurs when `g++` is not installed. You can install it with:

```
apt-get install g++
```

and run the installation again.

## `pyrevolve/schedulers/crevolve.cpp:25:10: fatal error: Python.h: No such file or directory` on Linux

This error occurs when the `python-dev` package was not installed. You can install it with:
```
apt-get install python3.10-dev
```
replace `3.10` with your installed python version.

And run the installation again.

## `error: legacy-install-failure` on MacOS

This error might occur when `brew` or `pip` are outdated.

Update brew:
```
brew update
```

Update pip:
```
pip install --upgrade pip
```

And run the installation again.

## `Error: Process completed with exit code 1.` when installing Stride on Windows

Unfortunately Stride can't be installed on a Windows platform, therefore NDK is also unsupported.

## `Error: Cannot install under Rosetta 2 in ARM default prefix (/opt/homebrew)!` on MacOS

This error occurs when you are not running the native homebrew for an ARM platform.
To proceed with the installation you can:

* [Migrate to native homebrew](https://blog.smittytone.net/2021/02/07/how-to-migrate-to-native-homebrew-on-an-m1-mac/)

or

* Prepend the brew install commands with `arch -arm64`:
```
arch -arm64 brew install ...
```

## Getting error `codepy.CompileError: module compilation failed`

This error occurs when the compiler wasn't able to perform the compilation, it can be caused by a environment configuration problem. Check the `DEVITO_ARCH` environment variable, it should be set with the compiler devito will use to compile the code.

You can find further information in the [Devito](https://github.com/devitocodes/devito/wiki/) documentation.

## Getting error `codepy.CompileError: module compilation failed` with `fatal error: 'omp.h' file not found`

This error occurs when the `libomp` is not installed or can not be found by the compiler.

Make sure to install it and export the environment variable `CPATH` with the path to the folder containing libomp headers.

## Getting error `ModuleNotFoundError: No module named 'neurotechdevkit'`

This error is shown when `neurotechdevkit` is not installed, if you installed it using a virtual environment like poetry you must run the script with `poetry run` or activate the environment.

## Getting error `AttributeError: module 'napari' has no attribute 'Viewer'` when calling `render_layout_3d`

This error is shown when napari is not installed, make sure to run

  `pip install "napari[all]"`

(or `pip install "napari[pyqt6_experimental]"` if running on a Mac M1)

and try again.
