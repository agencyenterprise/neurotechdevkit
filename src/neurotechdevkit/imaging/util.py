from typing import Mapping, Optional

import numpy as np
import numpy.typing as npt
import stride
from mosaic.types import Struct


def log_compress(iq_signals_beamformed, dynamic_range_db: float = 40):
    """Log-compress the beamformed IQ signals."""
    assert np.iscomplexobj(iq_signals_beamformed), "I/Q signals should be complex"

    real_envelope = np.abs(iq_signals_beamformed)
    real_envelope[real_envelope == 0] = np.finfo(float).eps  # add .eps for to allow log10
    image_db = 20 * np.log10(real_envelope / np.max(real_envelope)) + dynamic_range_db
    image_db = np.clip(image_db / dynamic_range_db, 0, None)

    assert ((0 <= image_db) & (image_db <= 1)).all(), "Expected values in range [0, 1]"
    # Range [0, 1] corresponds to [-dynamic_range_db dB, 0 dB]

    return image_db


def gamma_compress(iq_signals_beamformed, exponent: float = 0.5):
    """Gamma-compress the beamformed IQ signals."""
    assert np.iscomplexobj(iq_signals_beamformed), "I/Q signals should be complex"

    real_envelope = np.abs(iq_signals_beamformed)
    normalized = real_envelope / real_envelope.max()
    image_gamma = np.power(normalized, exponent)

    return image_gamma


def add_material_fields_to_problem(
    problem: stride.Problem,
    materials: Mapping[str, Struct],
    layer_ids: Mapping[str, int],
    masks: Mapping[str, npt.NDArray[np.bool_]],
    rng: Optional[np.random.Generator] = None,
) -> stride.Problem:
    """Add material fields as media to the problem.

    Included fields are:

    - the speed of sound (in m/s)
    - density (in kg/m^3)
    - absorption (in dB/cm)

    Args:
        problem (stride.Problem): the stride Problem object to which the
            media should be added.
        materials (Mapping[str, Struct]): a mapping from material names
            to Structs containing the material properties.
        layer_ids (Mapping[str, int]): a mapping from material names to
            integers representing the layer number for each material.
        masks (Mapping[str, npt.NDArray[np.bool_]]): a mapping from material
            names to boolean masks indicating the gridpoints.
    """
    grid = problem.grid

    vp = stride.ScalarField(name="vp", grid=grid)  # [m/s]
    rho = stride.ScalarField(name="rho", grid=grid)  # [kg/m^3]
    alpha = stride.ScalarField(name="alpha", grid=grid)  # [dB/cm]
    layer = stride.ScalarField(name="layer", grid=grid)  # integers

    rng = np.random.default_rng(rng)
    for name, material in materials.items():
        material_mask = masks[name]
        vp.data[material_mask] = material.vp
        if "vp_std" in material:
            vp_noise = rng.normal(scale=material.vp_std, size=material_mask.shape)
            vp.data[material_mask] += vp_noise[material_mask]
        rho.data[material_mask] = material.rho
        if "rho_std" in material:
            rho_noise = rng.normal(scale=material.rho_std, size=material_mask.shape)
            rho.data[material_mask] += rho_noise[material_mask]
        alpha.data[material_mask] = material.alpha
        if "alpha_std" in material:
            alpha_noise = rng.normal(scale=material.alpha_std, size=material_mask.shape)
            alpha.data[material_mask] += alpha_noise[material_mask]
        layer.data[material_mask] = layer_ids[name]

    vp.pad()
    rho.pad()
    alpha.pad()
    layer.pad()

    problem.medium.add(vp)
    problem.medium.add(rho)
    problem.medium.add(alpha)
    problem.medium.add(layer)

    return problem