import json

from fastapi.testclient import TestClient

from maze_agent.server import app, get_brain
from tests.test_agent import HintBrain

client = TestClient(app)


def test_levels_and_404():
    assert len(client.get("/api/levels").json()["levels"]) == 6
    assert client.get("/api/levels/6").json()["world"]["door"]
    assert client.get("/api/levels/7").status_code == 404


def test_graph_diagram():
    assert "think" in client.get("/api/graph").json()["mermaid"]


def test_play_streams_sse_until_done():
    app.dependency_overrides[get_brain] = HintBrain
    try:
        body = client.get("/api/levels/1/play").text
    finally:
        app.dependency_overrides.clear()
    events = [line.removeprefix("event: ") for line in body.splitlines() if line.startswith("event: ")]
    acts = [json.loads(line.removeprefix("data: ")) for line in body.splitlines() if '"type": "act"' in line]
    assert events[:2] == ["think", "act"] and events[-1] == "done"
    assert acts[-1]["status"] == "won"
