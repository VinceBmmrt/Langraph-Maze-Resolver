# Maze Agent

A LangGraph demo: an LLM-driven cat escapes 6 mazes of rising difficulty (walls, then spikes, dogs, and a locked door with its key).

![Level 4 played live by gpt-5.4-mini (sped up)](docs/demo.gif)

## Run

```bash
cp .env.example .env        # add your OPENAI_API_KEY
uv run uvicorn maze_agent.server:app
```

Open http://127.0.0.1:8000 and press Play. Levels chain automatically.

- Terminal only: `uv run python -m maze_agent --level 5`
- Tests (no API key needed): `uv run pytest`

## How it works

```
START -> think -> act -> think -> ... -> END
```

- `think` (async LLM node): turns what the cat sees into text and asks for a structured `Decision` (direction + thought). It has a node `timeout` and a `RetryPolicy`.
- `act` (pure rules): moves the cat, applies spikes, dogs, key and door, then routes with `Command(goto=...)` back to `think` or to `END`.
- **State** is a Pydantic model. Tokens are summed with a reducer (`Annotated[int, add]`).
- **Runtime context** (`context_schema`) carries the maze and the brain, so tests swap the LLM for a scripted brain.
- **Mental map**: from cells it already visited, the cat is told the nearest unexplored opening. Without it, small models walk in circles.
- **Checkpointer**: `InMemorySaver` with an explicit msgpack allowlist for the state types.
- **Streaming**: `graph.astream(stream_mode="updates")` is forwarded to the browser over FastAPI's native SSE. Closing the page cancels the run.

## Code

| File | Role |
|---|---|
| `maze_agent/maze.py` | seeded maze generation, path search |
| `maze_agent/world.py` | hazard rules (pure functions) |
| `maze_agent/levels.py` | the 6 levels, hazards placed so every level is solvable |
| `maze_agent/brain.py` | prompt, `Decision` schema, `LLMBrain` (`init_chat_model`) |
| `maze_agent/agent.py` | the LangGraph graph |
| `maze_agent/runner.py` | runs a level as JSON events |
| `maze_agent/server.py` | FastAPI + SSE |
| `maze_agent/__main__.py` | terminal runner |
| `maze_agent/config.py` | settings from `.env` |
| `static/` | canvas UI, live graph (Mermaid), thought feed |

## Config

`MODEL` in `.env` takes an `init_chat_model` id (default `openai:gpt-5.4-mini`). Another provider needs its package, e.g. `uv add langchain-anthropic` for `anthropic:claude-haiku-5-5`. Set `LANGSMITH_TRACING=true` to trace runs in LangSmith.
