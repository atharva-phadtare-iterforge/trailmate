# TrailMate: Status

## Day 1

- Set up the Python project using `uv` and FastAPI.
- Learned and implemented a basic LangGraph workflow.
- Created a graph with entry, processing, and response nodes.
- Added an API endpoint to run messages through the graph.
- Implemented a simple templated response without an LLM.

## Day 2

- Learned conditional routing and rule-based classification.
- Classified messages into `trail`, `general`, and `reject`.
- Added conditional edges for each category.
- Created separate templated responses for each category.
- Organized the hiking workflow using a separate FastAPI router.

## Day 3
- Learned about persistence, checkpoints, and stores in LangGraph.
- Got a small language model running locally using Ollama.
- Set up LiteLLM with a couple of named model tiers.
- Replaced one path’s templated response with a real model-generated response through the LiteLLM proxy.
- Kept the routing decision deterministic; only the text generation uses the model.
- Ensured at least one path in the graph returns a real, model-generated response while the other paths can remain templated.
- Added Streaming SSE to the API endpoint.
- Implemented token-by-token streaming so generated responses are delivered progressively to the client.