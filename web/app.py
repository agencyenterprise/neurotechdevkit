"""Flask app factory."""
import matplotlib
from flask import Flask
from matplotlib import rc


def create_app():
    """Create the Flask app."""
    matplotlib.use("Agg")

    rc("animation", html="html5")

    app = Flask(__name__, static_folder="static", template_folder="templates")

    # Register blueprints here
    from web.views import bp

    app.register_blueprint(bp)

    return app
