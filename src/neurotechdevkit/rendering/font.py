import pathlib

from matplotlib import font_manager

_FONT_DIR = pathlib.Path(__file__).parent / "fonts"


for font_file in _FONT_DIR.iterdir():
    if font_file.suffix.lower() != ".ttf":
        continue
    font_manager.fontManager.addfont(font_file)


TITLE_FONT_PROPS = font_manager.FontProperties(
    family="manrope",
    weight=800,
    size=20,
)

AXIS_LABEL_FONT_PROPS = font_manager.FontProperties(
    family="manrope",
    weight=800,
    size=14,
)


AXIS_TICK_FONT_PROPS = font_manager.FontProperties(
    family="manrope",
    weight=600,
    size=10,
)


LEGEND_FONT_PROPS = font_manager.FontProperties(
    family="manrope",
    weight=600,
    size=14,
)
