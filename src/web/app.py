"""Flask app factory."""
import matplotlib
from flask import Flask
from matplotlib import rc
from web.simulation_runner import SimulationRunner


def create_app():
    """Create the Flask app."""
    matplotlib.use("Agg")

    rc("animation", html="html5")

    app = Flask(__name__, static_folder="static", template_folder="templates")

    # Register blueprints here
    from web.views import bp

    app.register_blueprint(bp)

    SimulationRunner().initialize()

    return app


app = create_app()

def run():
    app.run(host='0.0.0.0', port=5000, debug=True)