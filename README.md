# Neurotech Development Kit (NDK) Research

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

Install `poetry` if it is not already on your system:

```bash
$ curl -sSL https://install.python-poetry.org | python -
```

Install research dependencies for the project either to the current virtual environment or a new one:

```bash
$ poetry install
```

Install stride with
```bash
$ poetry run pip install git+https://github.com/trustimaging/stride
```
---

This will also install the `neurotechdevkit` package.

If you are not using a virtual environment, `poetry` will [create one for you by default](https://python-poetry.org/docs/basic-usage/#using-your-virtual-environment). If you are already using a virtual environment, then `poetry` will install dependencies to that environment.


### Environment

If you let poetry manage your virtual environment, you can use the environment in one of two ways:

1. Activate the environment directly via:
   ```bash
   $ poetry shell
   ```
2. Prepend `poetry run` to any python command in order to run the command within the virtual environment. Example:
   ```bash
   $ poetry run foo
   ```

If you are already using your own virtual environment, you should not need to change anything.

### Development

For development, the installation instructions are the same as listed above (poetry installs developer requirements by default).

See our [contribution requirements](docs/contributing.md) for more information on how to contribute.

A Makefile is provided to assist with common commands such as linting and running unit tests.

```bash
$ make lint

$ make test
```

See the Makefile for other commands.

### Acknowledgements

Thanks to Fred Ehrsam for supporting this project, Quintin Frerichs and Milan Cvitkovic for providing direction, and to Sumner Norman for his ultrasound and neuroscience expertise. Thanks to [Stride](https://www.stride.codes/) for facilitating ultrasound simulations, [Devito](https://www.devitoproject.org/) for providing the backend solver, [Napari](https://napari.org/stable/) for great 3D visualization, and to [Jean-Francois Aubry, et al.](https://doi.org/10.1121/10.0013426) for the basis of the simulation scenarios.
