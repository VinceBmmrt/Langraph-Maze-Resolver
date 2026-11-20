# Maze Agent 🐈

English · [Français](README.fr.md)

A fun little project to play with LangGraph: an LLM-driven cat tries to escape 6 mazes, each harder than the last. Spikes, patrolling poodles, a locked door and its key stand in its way.

![The cat escaping level 3, its thoughts written live (sped up)](docs/demo.gif)

## Run it

```bash
cp .env.example .env        # put your OPENAI_API_KEY in it
uv run uvicorn maze_agent.server:app
```

Open http://127.0.0.1:8000 and press Play.

## What the cat can do

- **Think out loud**: each move comes with a short thought, streamed word by word.
- **Ask you for help**: when it goes round in circles, the game pauses and waits for your hint.
- **Learn from its failures**: after a loss it writes itself a lesson and reads it on the next try.
- **Run on different models**: GPT-5.4 mini, GPT-5.4 nano or Claude Haiku, with the cost of each run shown live.
- **Get benchmarked**: `uv run python -m maze_agent.eval` lets the models race on every level (see the Evaluation tab).

## The LangGraph bits

```
START -> think -> act -> think ...           explore
                      -> ask_human -> think  stuck: wait for the player
                      -> reflect -> END      lost: write a lesson
                      -> END                 escaped
```

- `StateGraph` with a Pydantic state, and reducers to add up tokens and cost
- `Command(goto=...)` so the `act` node picks what comes next
- `interrupt()` and `Command(resume=...)` for the call for help, on top of a checkpointer
- the `Store` as long-term memory for the lessons
- `stream_mode=["updates", "messages"]` to move the cat and stream its thoughts
- `context_schema` to plug in the LLM (tests use a scripted cat, no API key needed)
- node `timeout` and `RetryPolicy` on LLM calls
- `draw_mermaid()` to show the live graph in the page

The graph lives in `maze_agent/agent.py`. Tests: `uv run pytest`.
