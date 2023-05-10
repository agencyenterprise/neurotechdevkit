This page contains a list of known problems you might face when installing and running NDK and the actions to solve them.


### `Error: Process completed with exit code 1.` when installing Stride on Windows

Unfortunately Stride can't be installed on a Windows platform, therefore NDK is also unsupported.

### Getting error `codepy.CompileError: module compilation failed`

This error occurs when the compiler wasn't able to perform the compilation, it can be caused by a environment configuration problem. Check the `DEVITO_ARCH` environment variable, it should be set with the compiler devito will use to compile the code.

You can find further information in the [Devito](https://github.com/devitocodes/devito/wiki/) documentation.

### Getting error `codepy.CompileError: module compilation failed` with `fatal error: 'omp.h' file not found`

This error occurs when the `libomp` is not installed or can not be found by the compiler.

Make sure to install it and export the environment variable `CPATH` with the path to the folder containing libomp headers.

### Getting error `ModuleNotFoundError: No module named 'neurotechdevkit'`

This error is shown when `neurotechdevkit` is not installed, if you installed it using a virtual environment like poetry you must run the script with `poetry run` or activate the environment.

### Getting error `AttributeError: module 'napari' has no attribute 'Viewer'` when calling `render_layout_3d`

This error is shown when napari is not installed, make sure to run

  `poetry run pip install "napari[all]"`

and try again.
