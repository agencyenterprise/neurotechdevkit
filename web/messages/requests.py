"""Request models for the web messages sent and received by the web app."""
from web.messages.settings import DefaultSettings


class RenderLayoutRequest(DefaultSettings):
    """Render layout request model for the render layout request."""

    pass


class SimulateRequest(DefaultSettings):
    """Simulate request model for the simulate request."""

    pass


class IndexBuiltInScenario(DefaultSettings):
    """The Scenario settings for the index page."""

    pass
