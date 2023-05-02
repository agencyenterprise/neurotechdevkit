from functools import wraps
from typing import Callable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from IPython.display import Video
from matplotlib.animation import FFMpegWriter, FuncAnimation

ARTIST_NAME = "NDK Research"


def display_video_file(file_name: str) -> Video:
    """Renders a video file in a Ipython environment.

    Args:
        file_name: the file name containing the animation to display.

    Returns:
        A video object.
    """
    return Video(file_name)


def video_only_output(func: Callable) -> Callable:
    """A decorator to create the context for video creation.

    It deactivates the interactive environment while the animation is being created.
    It re-activates the interactive environment after the animation was created.

    It temporarily modifies the size of the `animation.embed_limit`. Changing
    `animation.embed_limit` is required to have larger animations embedded in
    the notebook without the need of writing to disk.

    Args:
        func: a function that creates an animation.

    Returns:
        A decorated function.
    """

    @wraps(func)
    def inner(*args, **kwargs):
        embed_limit = matplotlib.rcParams.get("animation.embed_limit", None)
        interactive = plt.isinteractive()
        plt.ioff()
        video = func(*args, **kwargs)
        plt.close()
        if interactive:
            plt.ion()
        matplotlib.rcParams["animation.embed_limit"] = embed_limit
        return video

    return inner


def configure_matplotlib_for_embedded_animation() -> None:
    """Set up matplotlib parameters for correct visualization of animations in notebook.

    The `matplotlib.rcParams['animation.html']` will be changed to 'jshtml`. Using
    javascript html (jshtml) as backend waives the requirement to have installed
    `ffmpeg` to visualize animations in a notebook. It also enables interactive
    functionality for the animation.

    Changing `animation.embed_limit` is required to have larger animations embedded in
    the notebook without the need of writing to disk.
    """
    plt.rcParams["animation.embed_limit"] = 2**100
    plt.rcParams["animation.html"] = "jshtml"


def make_animation(
    fig: plt.Figure,
    ax: plt.Axes,
    wavefield: npt.NDArray[np.float_],
    n_frames_undersampling: int,
) -> FuncAnimation:
    """Creates an animation of a time evolution of `wavefield`.

    Args:
        fig: matplotlib figure that would act as template for the animation.
        ax: matplotlib figure axes where the actual wavelet is displayed.
        wavefield: the wavefield of the simulation.
        n_frames_undersampling: the number of wavefield time steps to skip during the
            creation of the animation.

    Returns:
        An animation object.
    """

    if wavefield.ndim != 3:
        raise ValueError("Animations only supported for 2D scenarios/slices only.")

    wavefield = wavefield[:, :, ::n_frames_undersampling].copy()

    # if `wavefield`` contain only very small numbers, particularly when selecting
    # slices of 3D simulations, matplotlib occasionally throws an overflow error
    # it can be solved rounding very small numbers to zero. Note that this only
    # affects the visualization
    wavefield = np.round(wavefield, 5)

    im = ax.get_images()[0]
    im.set_array(wavefield[:, :, 0])

    wavefield = wavefield[:, :, 1:]
    n_frames = wavefield.shape[-1]

    def init():
        return [im]

    def animate(i):
        data = wavefield[:, :, i]
        im.set_array(data)
        return [im]

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=n_frames,
        repeat=False,
        save_count=n_frames,
        blit=True,
    )

    return anim


def save_animation(
    animation: FuncAnimation,
    file_name: str,
    fps: int = 25,
    dpi: int = 100,
    bitrate: int = 2500,
) -> None:
    """
    Saves an animation object to a file in disk.

    `ffmpeg` is required to create the animation.
    Currently only mp4 format is supported.

    Args:
        animation: an animation object.
        file_name: the file name (path included and format) where the animation will be
                saved.
        fps: the number of frames per second to show during the animation.
        dpi: dots per inch for the resulting animation.
        bitrate: bitrate for the saved movie file (controls the file size and quality).
    """
    FFwriter = FFMpegWriter(
        fps=fps, codec="h264", metadata=dict(artist=ARTIST_NAME), bitrate=bitrate
    )
    animation.save(file_name, writer=FFwriter, dpi=dpi)
