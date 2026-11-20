import json
import os

from fastapi.testclient import TestClient

from maze_agent.server import app, brain_factory
from tests.test_agent import FakeBrain, HintBrain

os.environ.setdefault("OPENAI_API_KEY", "test-key")  # CI has no key; the brain is scripted anyway
client = TestClient(app)
MODEL = "openai:gpt-5.4-mini"


def sse(body: str) -> list[tuple[str, dict]]:
    events, name = [], None
    for line in body.splitlines():
        if line.startswith("event: "):
            name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            events.append((name, json.loads(line.removeprefix("data: "))))
    return events


def stream(url: str, brain) -> list[tuple[str, dict]]:
    app.dependency_overrides[brain_factory] = lambda: lambda _model: brain
    try:
        return sse(client.get(url).text)
    finally:
        app.dependency_overrides.clear()


def test_levels_models_and_404():
    info = client.get("/api/levels").json()
    assert len(info["levels"]) == 6 and any(m["id"] == MODEL for m in info["models"])
    assert client.get("/api/levels/6").json()["world"]["door"]
    assert client.get("/api/levels/7").status_code == 404
    assert client.get("/api/levels/1/play?model=nope").status_code == 400


def test_graph_diagram_shows_every_node():
    mermaid = client.get("/api/graph").json()["mermaid"]
    assert all(node in mermaid for node in ("think", "act", "ask_human", "reflect"))


def test_play_streams_sse_until_done():
    events = stream(f"/api/levels/1/play?model={MODEL}", HintBrain())
    names = [name for name, _ in events]
    assert names[:3] == ["start", "think", "act"] and names[-1] == "done"
    assert [data for name, data in events if name == "act"][-1]["status"] == "won"


def test_interrupt_then_resume_over_http():
    brain = FakeBrain()  # circles on the start cell, so it asks for help
    events = stream(f"/api/levels/1/play?model={MODEL}", brain)
    thread_id = events[0][1]["thread_id"]
    assert events[-2][0] == "interrupt"
    resumed = stream(f"/api/runs/{thread_id}/resume?answer=try+south", brain)
    assert resumed[0] == ("ask_human", {"type": "ask_human", "advice": "try south", "asked_at": 8})
