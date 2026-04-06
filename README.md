---
title: Customer Support Triage OpenEnv
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
---

# Customer Support Triage Environment

A real-world task environment for OpenEnv focused on Customer Support Triage.

Processing support tickets is a genuine task that human support agents and AIs do daily.
The environment evaluates an AI agent's ability to read tickets from a queue, classify
and route them to the right departments, or directly reply to simple FAQ questions using
an internal Knowledge Base.

## Task Details

The agent is exposed to a queue of incoming customer support tickets.
The agent has access to a Knowledge Base with common FAQs like business hours or documentation links.

Tasks Available:
- easy: Single ticket requiring simple routing to billing.
- medium: 3 tickets requiring routing to billing, technical, and sales.
- hard: 5 tickets mixing routing and direct FAQ replies.

Set the environment variable SUPPORT_TRIAGE_TASK to easy, medium, or hard.

## Observation Space

SupportTriageObservation:
- message (str): System response of the previous action.
- remaining_tickets (int): Number of tickets left in the queue.
- ticket_queue (List[Dict]): List of tickets with ID and preview. Full text shown after read_ticket.
- knowledge_base (str): Internal KB snippets for FAQ answering.

## Action Space

SupportTriageAction:
- action_type (str): One of "read_ticket", "route_ticket", "reply_ticket", "done".
- ticket_id (int, optional): The ID of the ticket.
- department (str, optional): Routing destination - billing, technical, or sales.
- reply_text (str, optional): Message content to reply with.

## Grader and Reward

For each correctly processed ticket a partial reward of 1.0 / total_tickets is given.
An incorrect action gives 0.0 reward. Fully clearing the queue correctly gives score 1.0.

## Setup and Usage

Local testing:
    uv run server

Using with openenv-core:
    from support_triage.client import SupportTriageEnv
    env = SupportTriageEnv(base_url="http://localhost:8000")
    obs = env.reset()

## Baseline Inference

    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4o-mini"
    export HF_TOKEN="your_key_here"
    export SUPPORT_TRIAGE_TASK="hard"
    python inference.py

Expected output:
    [START] task=hard env=support_triage model=gpt-4o-mini
    [STEP] step=1 action={"action_type": "route_ticket", ...} reward=0.20 done=false error=null
    [END] success=true steps=5 score=1.000 rewards=0.20,0.20,0.20,0.20,0.20
