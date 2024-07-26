"""Collection of smoke tests for the web app."""
import json

import pytest
from web.app import app


@pytest.fixture
def test_client(tmp_path):
    """Create a test client for the flask app."""
    app.config["CT_FOLDER"] = tmp_path
    with app.test_client() as testing_client:
        yield testing_client


def test_index(test_client):
    """Test the index page."""
    response = test_client.get("/")
    assert response.status_code == 200


def test_render_layout(test_client, request_payload_example):
    """Test the render_layout endpoint with the simplest payload."""
    response = test_client.post(
        "/render_layout",
        data=json.dumps(request_payload_example),
        content_type="application/json",
    )
    assert "data" in response.json
    assert len(response.json["data"]) == 48192
    assert response.status_code == 200


def test_invalid_payload_render_layout(test_client):
    """/render_layout returns a 400 Bad Request when given an invalid payload."""
    response = test_client.post(
        "/render_layout",
        data=json.dumps({"INCOMPLETE": "PAYLOAD"}),
        content_type="application/json",
    )
    assert response.status_code == 400


@pytest.mark.integration
def test_simulate(test_client, request_payload_example):
    """Test the simulate endpoint with the simplest payload."""
    response = test_client.post(
        "/simulation",
        data=json.dumps(request_payload_example),
        content_type="application/json",
    )
    assert "data" in response.json
    assert len(response.json["data"]) == 80936
    assert response.status_code == 200


def test_invalid_payload_simulate(test_client):
    """/simulate returns a 400 Bad Request when given an invalid payload."""
    response = test_client.post(
        "/simulation",
        data=json.dumps({"INCOMPLETE": "PAYLOAD"}),
        content_type="application/json",
    )
    assert response.status_code == 400
