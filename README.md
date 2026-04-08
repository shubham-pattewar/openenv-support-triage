---
title: Customer Support Triage OpenEnv
emoji: "📨"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---

# Customer Support Triage OpenEnv

`support_triage` is a real-world OpenEnv benchmark for customer support operations. The agent works through an inbox of incoming tickets, chooses when to inspect full ticket text, routes operational issues to the correct team, and answers simple FAQ tickets from an internal knowledge base.

This is meant to model actual work that human support specialists and support copilots do every day:

- triaging billing, technical, and sales requests
- avoiding misroutes on ambiguous tickets
- giving accurate FAQ replies from internal documentation
- managing a queue efficiently under a step budget

## Why this environment is useful

Most agent benchmarks over-index on browser control or toy tasks. Support triage is a high-value operational workflow with clear success criteria, partial-progress rewards, and genuine failure modes:

- correct action choice matters
- reading before acting can improve outcomes
- rushed routing can still get partial credit on easy tickets but loses points on ambiguous cases
- invalid actions and premature episode termination reduce final score

## OpenEnv Interface

The environment implements the standard OpenEnv API:

- `reset(task="easy" | "medium" | "hard") -> observation`
- `step(action) -> observation, reward, done, info`
- `state() -> current episode state`

Typed Pydantic models are defined in `models.py`.

### Action space

`SupportTriageAction`

- `action_type`: `"read_ticket" | "route_ticket" | "reply_ticket" | "done"`
- `ticket_id`: integer ticket ID for read/route/reply actions
- `department`: `"billing" | "technical" | "sales"` for routing
- `reply_text`: customer-facing response for FAQ tickets

### Observation space

`SupportTriageObservation`

- `task_id`: current task split
- `task_objective`: natural-language episode goal
- `message`: environment feedback after the last action
- `remaining_tickets`: unresolved queue size
- `ticket_queue`: visible queue state; unread tickets expose previews, read tickets expose full text
- `processed_ticket_ids`: tickets already handled
- `knowledge_base`: internal support KB available to the agent
- `last_action_error`: validation or execution error from the previous action
- `grader_score`: final task score in `[0.0, 1.0]` when the episode ends

## Tasks

The benchmark contains three deterministic tasks with increasing difficulty.

### Easy

- 3 tickets
- straightforward billing, technical, and FAQ tickets
- rewards fast, correct routing and correct business-hours reply

### Medium

- 4 tickets
- mixes clear tickets with one deceptive routing case
- includes an API documentation reply ticket

### Hard

- 6 tickets
- multiple ambiguous previews
- multiple FAQ replies
- penalizes premature `done`, invalid actions, and sloppy triage under a fixed step budget

## Reward design

The reward function is shaped over the full trajectory.

- reading a ticket gives a small positive reward because it reveals information
- correctly resolving a ticket increases the normalized episode score
- ambiguous tickets lose credit if routed without being read first
- incorrect actions give `0.0` step reward and reduce the final trajectory score through deterministic penalties
- final `grader_score` is clipped to `[0.0, 1.0]`

This gives dense enough feedback for learning while preserving a clear end-task objective.

## Project layout

```text
.
├── client.py
├── graders.py
├── inference.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── Dockerfile
└── server
    ├── app.py
    ├── Dockerfile
    └── support_triage_environment.py
```

## Local setup

Create the environment and install dependencies:

```bash
uv sync
```

Run the API locally:

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Validate the environment:

```bash
uv run openenv validate
```

## Docker

Build and run from the repository root:

```bash
docker build -t support-triage-openenv .
docker run --rm -p 8000:8000 support-triage-openenv
```

The container serves the OpenEnv API on port `8000`.

## Hugging Face Spaces

This repository is structured for a Docker Space deployment.

- SDK: `docker`
- App port: `8000`
- Tag the Space with `openenv`

Once deployed, the validator should be able to `POST /reset` successfully.

## Baseline inference

The required root-level baseline script is [`inference.py`](./inference.py). It:

- uses the OpenAI client for all model calls
- reads `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- optionally uses `LOCAL_IMAGE_NAME` when running against a local Docker image
- runs all three tasks sequentially
- emits `[START]`, `[STEP]`, and `[END]` lines in the required format

Example environment variables:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token"
python inference.py
```

Reference baseline targets with a competent instruction-tuned model and `temperature=0.0`:

- easy: `0.85+`
- medium: `0.70+`
- hard: `0.55+`

Exact results depend on the served model, but the tasks and grader are deterministic.

## Validation helper

You can run the included submission validator after deployment:

```bash
bash validate-submission.sh https://your-space-name.hf.space .
```
