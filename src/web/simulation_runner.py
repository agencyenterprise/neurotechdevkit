"""SimulationRunner class, which is a singleton that manages the simulation process."""
import asyncio
import ctypes
from queue import Queue
from threading import Thread
from typing import Awaitable, Optional, Tuple


def terminate_thread(thread: Thread):
    """Terminates a python thread from another thread.

    Args:
        thread: The thread to terminate.
    """
    assert thread.ident is not None, "thread ID must be defined"
    exc = ctypes.py_object(SystemExit)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread.ident), exc)
    if res == 0:
        return
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread.ident, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


class Response:
    """Base class for responses."""

    type: str


class SimulationResponse(Response):
    """Response containing the result of the simulation."""

    type = "simulation"
    data: Tuple[str, str]

    def __init__(self, data: Tuple[str, str]):
        """Initialize the response.

        Args:
            data: The result of the simulation.
        """
        self.data = data


class NoSimulationResponse(Response):
    """Response indicating that the simulation has not been run."""

    type = "no_simulation"


class SimulationRunner(object):
    """Runs the simulation process and stores the result."""

    _running: bool
    _last_result: Response
    _requests_queue: Queue
    _configuration: dict

    def __new__(cls):
        """Singleton implementation."""
        if not hasattr(cls, "instance"):
            cls.instance = super(SimulationRunner, cls).__new__(cls)
        return cls.instance

    def get(self) -> Response:
        """Get the stored result or wait for the simulation to finish."""
        if self.has_last_result:
            return self._last_result

        if self._running:
            callback_queue: Queue[Response] = Queue(1)
            self._requests_queue.put(callback_queue)
            return callback_queue.get()
        return NoSimulationResponse()

    def _run_in_loop(self, loop, executor, result):
        asyncio.set_event_loop(loop)
        result.append(loop.run_until_complete(executor))

    def run(self, executor: Awaitable, configuration: dict) -> Response:
        """Run the simulation process and store the result.

        Args:
            executor: The coroutine to run.
            configuration: The configuration of the simulation.

        Returns:
            The result of the simulation.
        """
        self._running = True
        self._configuration = configuration
        communication_pipe: list[Tuple[str, str]] = []
        self._thread = Thread(
            target=self._run_in_loop, args=(self._loop, executor, communication_pipe)
        )
        self._thread.start()
        self._thread.join()
        self._running = False
        result: Response
        if len(communication_pipe) > 0:
            result = SimulationResponse(communication_pipe.pop())
        else:
            result = NoSimulationResponse()
        self._last_result = result

        while not self._requests_queue.empty():
            callback = self._requests_queue.get()
            callback.put(result)
            callback.task_done()
        return result

    @property
    def is_running(self) -> bool:
        """Whether the simulation is running."""
        return self._running

    @property
    def has_last_result(self) -> bool:
        """Whether the simulation has a result."""
        return isinstance(self._last_result, SimulationResponse)

    @property
    def configuration(self) -> Optional[dict]:
        """The configuration of the simulation.

        Returns:
            The configuration of the simulation if it exists, None otherwise.
        """
        if self._configuration:
            return self._configuration
        return None

    def initialize(self):
        """Initialize the simulation runner."""
        self._requests_queue = Queue()
        self._last_result = None
        self._thread = None
        self._loop = asyncio.new_event_loop()
        self._running = False
        self._configuration = None

    def reset(self):
        """Reset the simulation runner."""
        if self._thread:
            terminate_thread(self._thread)
            self._thread = None
        self._loop = asyncio.new_event_loop()
        self._running = False
        self._configuration = None
        self._last_result = NoSimulationResponse()
