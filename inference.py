import asyncio
import os
import textwrap
import json
from typing import List, Optional

from openai import OpenAI

from models import SupportTriageAction
from client import SupportTriageEnv

IMAGE_NAME = os.getenv("IMAGE_NAME")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "dummy-key")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
TASK_NAME = os.getenv("SUPPORT_TRIAGE_TASK", "easy")
BENCHMARK = os.getenv("SUPPORT_TRIAGE_BENCHMARK", "support_triage")
MAX_STEPS = 15
TEMPERATURE = 0.0

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an AI Customer Support Agent. You must process customer support tickets.
    You will be given the system Observation containing a list of `ticket_queue` waiting for you, and a `knowledge_base` with FAQs.
    
    You can take the following actions:
    1. "read_ticket": Read the full text of a specific ticket. (You must pass `ticket_id`)
    2. "route_ticket": Route a ticket to the right department. Valid departments are "billing", "technical", "sales". (You must pass `ticket_id` and `department`)
    3. "reply_ticket": Reply to a ticket that is just asking a FAQ question. Only reply to clear FAQ questions using info from the knowledge base. (You must pass `ticket_id` and `reply_text`)
    4. "done": Call this when the ticket queue is completely empty.

    Rules:
    - You must output exactly one valid JSON object representing your Action. 
    - The JSON object must contain `action_type`. It optionally contains `ticket_id` (integer), `department`, and `reply_text`.
    - Do not wrap the JSON in Markdown formatting like ```json ... ```. Just output the raw JSON.
    - Example 1: {"action_type": "read_ticket", "ticket_id": 101}
    - Example 2: {"action_type": "route_ticket", "ticket_id": 101, "department": "billing"}
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, obs_msg: str, tickets: list, queue_count: int, kb: str, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Knowledge Base: {kb}
        Remaining Tickets count: {queue_count}
        Ticket Queue: {json.dumps(tickets, indent=2)}
        System Message from last action: {obs_msg}
        
        Previous actions:
        {history_block}
        
        Based on this, what is your next action? Reply in valid raw JSON.
        """
    ).strip()


def get_model_action(client: OpenAI, step: int, obs_msg: str, tickets: list, queue_count: int, kb: str, history: List[str]) -> SupportTriageAction:
    user_prompt = build_user_prompt(step, obs_msg, tickets, queue_count, kb, history)
    
    # Simple fallback action string if JSON fails to parse
    action_str = '{"action_type": "read_ticket", "ticket_id": null}'
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            response_format={"type": "json_object"},
        )
        action_str = (completion.choices[0].message.content or "").strip()
        # Remove any Markdown code block wrapping if LLM gets confused
        if action_str.startswith("```json"): action_str = action_str[7:]
        if action_str.startswith("```"): action_str = action_str[3:]
        if action_str.endswith("```"): action_str = action_str[:-3]
        action_str = action_str.strip()
        
        parsed = json.loads(action_str)
        return SupportTriageAction(**parsed), action_str
    except Exception as exc:
        with open("debug.log", "a") as f:
            f.write(f"Exception: {exc}\nRaw Action Str: {action_str}\n---\n")
        print(f"[DEBUG] Model request or parsing failed: {exc} | Raw string: {action_str}", flush=True)
        return SupportTriageAction(action_type="done"), action_str


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    try:
        if IMAGE_NAME:
            env = await SupportTriageEnv.from_docker_image(IMAGE_NAME)
        else:
            # Fallback for local testing (needs `uv run server.app`)
            env = SupportTriageEnv(base_url="http://localhost:8000")
    except Exception as e:
        print(f"[DEBUG] Failed to connect to env: {e}", flush=True)
        return

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
            
        last_msg = result.observation.message
        last_tickets = result.observation.ticket_queue
        last_count = result.observation.remaining_tickets
        kb = result.observation.knowledge_base

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action, raw_str = get_model_action(client, step, last_msg, last_tickets, last_count, kb, history)

            try:
                result = await env.step(action)
                obs = result.observation
                reward = result.reward or 0.0
                done = result.done
                error = None
            except Exception as e:
                reward = 0.0
                done = False
                error = str(e)
                obs = result.observation if result else None
                # break loop if catastrophic
                break

            rewards.append(reward)
            steps_taken = step
            
            # Escape action string for standard log formatting (no newlines)
            safe_action_str = raw_str.replace('\n', ' ').replace('\r', '')
            log_step(step=step, action=safe_action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {safe_action_str} -> reward {reward:+.2f}")

            if obs:
                last_msg = obs.message
                last_tickets = obs.ticket_queue
                last_count = obs.remaining_tickets
                kb = obs.knowledge_base

            if done:
                break

        score = sum(rewards)
        score = min(max(score, 0.0), 1.0)  # clamp to [0, 1]
        success = score >= 0.99 # almost perfect required

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())
