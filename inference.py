import asyncio
import json
import os
import textwrap
from typing import List, Optional, Tuple

from openai import OpenAI

from client import SupportTriageEnv
from models import SupportTriageAction

LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN")
BENCHMARK = "support_triage"
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 12
TEMPERATURE = 0.0
MAX_TOKENS = 220

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are operating a customer support triage desk.
    Choose exactly one next action as a raw JSON object.

    Available actions:
    - {"action_type":"read_ticket","ticket_id":101}
    - {"action_type":"route_ticket","ticket_id":101,"department":"billing"}
    - {"action_type":"reply_ticket","ticket_id":104,"reply_text":"..."}
    - {"action_type":"done"}

    Policies:
    - Ambiguous tickets should be read before routing.
    - Reply only to FAQ-style tickets using the knowledge base facts.
    - Route charge/refund/invoice issues to billing.
    - Route crashes/errors/bugs to technical.
    - Route pricing/pilot/procurement requests to sales.
    - Return raw JSON only, with no markdown or explanation.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(task_name: str, step: int, observation) -> str:
    return textwrap.dedent(
        f"""
        Task: {task_name}
        Objective: {observation.task_objective}
        Step: {step}
        Message: {observation.message}
        Remaining tickets: {observation.remaining_tickets}
        Processed tickets: {observation.processed_ticket_ids}
        Ticket queue:
        {json.dumps(observation.ticket_queue, ensure_ascii=True)}
        Knowledge base:
        {observation.knowledge_base}
        """
    ).strip()


def get_model_action(client: OpenAI, task_name: str, step: int, observation) -> Tuple[SupportTriageAction, str]:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(task_name, step, observation)},
        ],
    )
    raw = (response.choices[0].message.content or "").strip()
    payload = json.loads(raw)
    return SupportTriageAction(**payload), json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


async def create_env() -> SupportTriageEnv:
    if LOCAL_IMAGE_NAME:
        return await SupportTriageEnv.from_docker_image(LOCAL_IMAGE_NAME)
    return SupportTriageEnv(base_url="http://localhost:8000")


async def run_task(client: OpenAI, task_name: str) -> float:
    env = await create_env()
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task=task_name)
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action, action_str = get_model_action(client, task_name, step, result.observation)
            error: Optional[str] = None

            try:
                result = await env.step(action)
            except Exception as exc:
                error = str(exc).replace("\n", " ")
                log_step(step=step, action=action_str, reward=0.0, done=False, error=error)
                break

            reward = float(result.reward or 0.0)
            rewards.append(reward)
            steps_taken = step
            error = result.observation.last_action_error
            log_step(
                step=step,
                action=action_str,
                reward=reward,
                done=bool(result.done),
                error=error,
            )
            if result.done:
                score = float(result.observation.grader_score or 0.0)
                success = score >= 0.8
                break

        if not result.done:
            score = float(result.observation.grader_score or 0.0)
            success = score >= 0.8

    finally:
        try:
            await env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


async def main() -> None:
    if not API_KEY:
        raise RuntimeError("HF_TOKEN must be set for OpenAI client calls.")

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_name in TASKS:
        await run_task(client, task_name)


if __name__ == "__main__":
    asyncio.run(main())
