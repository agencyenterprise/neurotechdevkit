"""SimulationRunner class, which is a singleton that manages the simulation process."""
import asyncio
from queue import Queue
from typing import Awaitable, Optional, Tuple


class SimulationRunner(object):
    """Runs the simulation process and stores the result."""

    _running: bool
    _last_result: Tuple[str, str]
    _requests_queue: Queue
    _configuration: dict

    def __new__(cls):
        """Singleton implementation."""
        if not hasattr(cls, "instance"):
            cls.instance = super(SimulationRunner, cls).__new__(cls)
        return cls.instance

    def get(self) -> Tuple[str, str]:
        """Get the stored result or wait for the simulation to finish."""
        if self._last_result:
            return self._last_result

        if self._running:
            callback_queue = Queue(1)
            self._requests_queue.put(callback_queue)
            return callback_queue.get()

    def run(self, executor: Awaitable, configuration: dict) -> Tuple[str, str]:
        """Run the simulation process and store the result.

        Args:
            executor: The coroutine to run.
            configuration: The configuration of the simulation.

        Returns:
            The result of the simulation.
        """
        self._running = True
        self._configuration = configuration
        result = self._loop.run_until_complete(executor)
        self._last_result = result
        self._running = False
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
        return self._last_result is not None

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
        # TODO: stop the loop
        # if hasattr(self, '_loop') and self._loop:
        #     self._loop.stop()
        #     self._loop.close()
        self._loop = asyncio.new_event_loop()
        self._requests_queue = Queue()
        self._running = False
        self._last_result = None
        self._configuration = None
