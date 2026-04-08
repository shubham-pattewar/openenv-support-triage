"""HTTP client for the customer support triage environment."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import SupportTriageAction, SupportTriageObservation
except ImportError:
    from models import SupportTriageAction, SupportTriageObservation


class SupportTriageEnv(EnvClient[SupportTriageAction, SupportTriageObservation, State]):
    """Client wrapper for local or containerized execution."""

    def _step_payload(self, action: SupportTriageAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[SupportTriageObservation]:
        obs_data = payload.get("observation", {})
        observation = SupportTriageObservation(
            task_id=obs_data.get("task_id", "easy"),
            task_objective=obs_data.get("task_objective", ""),
            message=obs_data.get("message", ""),
            remaining_tickets=obs_data.get("remaining_tickets", 0),
            ticket_queue=obs_data.get("ticket_queue", []),
            processed_ticket_ids=obs_data.get("processed_ticket_ids", []),
            knowledge_base=obs_data.get("knowledge_base", ""),
            last_action_error=obs_data.get("last_action_error"),
            grader_score=obs_data.get("grader_score"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
