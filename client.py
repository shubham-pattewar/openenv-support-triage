from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import SupportTriageAction, SupportTriageObservation
except ImportError:
    from models import SupportTriageAction, SupportTriageObservation


class SupportTriageEnv(
    EnvClient[SupportTriageAction, SupportTriageObservation, State]
):
    """Client for the Support Triage Environment."""

    def _step_payload(self, action: SupportTriageAction) -> Dict:
        return {
            "action_type": action.action_type,
            "ticket_id": action.ticket_id,
            "department": action.department,
            "reply_text": action.reply_text,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SupportTriageObservation]:
        obs_data = payload.get("observation", {})
        observation = SupportTriageObservation(
            message=obs_data.get("message", ""),
            remaining_tickets=obs_data.get("remaining_tickets", 0),
            ticket_queue=obs_data.get("ticket_queue", []),
            knowledge_base=obs_data.get("knowledge_base", ""),
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
