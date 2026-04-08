"""Typed models for the customer support triage OpenEnv environment."""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class SupportTriageAction(Action):
    """Action schema for a support operations agent."""

    action_type: Literal["read_ticket", "route_ticket", "reply_ticket", "done"] = Field(
        ...,
        description=(
            "Action to perform. Read reveals full ticket text, route assigns a ticket to a "
            "department, reply resolves FAQ tickets, and done ends the episode."
        ),
    )
    ticket_id: Optional[int] = Field(
        default=None,
        description="Ticket identifier for read, route, and reply actions.",
    )
    department: Optional[Literal["billing", "technical", "sales"]] = Field(
        default=None,
        description="Destination team for route_ticket.",
    )
    reply_text: Optional[str] = Field(
        default=None,
        description="Customer-facing reply for reply_ticket.",
    )


class SupportTriageObservation(Observation):
    """Observation schema returned after each environment transition."""

    task_id: str = Field(..., description="Current task split: easy, medium, or hard.")
    task_objective: str = Field(..., description="High-level goal for the episode.")
    message: str = Field(..., description="Environment feedback for the most recent action.")
    remaining_tickets: int = Field(..., description="Number of unresolved tickets left in the queue.")
    ticket_queue: List[Dict[str, object]] = Field(
        ...,
        description="Visible ticket queue. Unread tickets show previews; read tickets show full text.",
    )
    processed_ticket_ids: List[int] = Field(
        ...,
        description="Ticket IDs already resolved during this episode.",
    )
    knowledge_base: str = Field(
        ...,
        description="Internal help-center snippets available to the agent.",
    )
    last_action_error: Optional[str] = Field(
        default=None,
        description="Validation or execution error for the most recent action, if any.",
    )
    grader_score: Optional[float] = Field(
        default=None,
        description="Final deterministic task score in the range [0.0, 1.0]. Present when done.",
    )
