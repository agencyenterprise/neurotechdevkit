from neurotechdevkit.grid import Grid


def test_instantiate_grid():
    """Validates the grid can be instantiated."""
    grid = Grid.make_grid(
        extent=(0.02, 0.03),  # 20cm x 30cm
        speed_water=1500.0,
        center_frequency=5e5,
        ppw=6,
    )
    assert grid.space.shape == (41, 61)
    assert grid.space.spacing == (0.0005, 0.0005)
    assert grid.time is None
