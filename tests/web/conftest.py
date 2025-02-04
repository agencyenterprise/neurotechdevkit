import pytest


@pytest.fixture()
def request_payload_example():
    return {
        "is2d": True,
        "displaySettings": None,
        "scenarioSettings": {
            "isPreBuilt": True,
            "scenarioId": "Scenario0",
        },
        "transducers": [
            {
                "aperture": 0.01,
                "delay": 0,
                "direction": [1, 0],
                "focalLength": 0.01,
                "numPoints": 1000,
                "position": [0.01, 0],
                "transducerType": "focusedSource",
            }
        ],
        "target": {
            "centerX": "0.0024",
            "centerY": "0.0285",
            "centerZ": None,
            "radius": "0.0017",
        },
        "simulationSettings": {
            "materialProperties": {
                "water": {
                    "vp": "1500",
                    "rho": "1000",
                    "alpha": "0",
                    "renderColor": "#2e86ab",
                },
                "brain": {
                    "vp": "1560",
                    "rho": "1040",
                    "alpha": "0.3",
                    "renderColor": "#db504a",
                },
                "corticalBone": {
                    "vp": "2800",
                    "rho": "1850",
                    "alpha": "4",
                    "renderColor": "#faf0ca",
                },
                "tumor": {
                    "vp": "1650",
                    "rho": "1150",
                    "alpha": "0.8",
                    "renderColor": "#94332f",
                },
            },
            "simulationPrecision": "5",
            "centerFrequency": "500000",
            "isSteadySimulation": True,
        },
    }
