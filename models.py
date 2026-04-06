# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Support Triage Environment.
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field
from typing import Literal, Optional, List, Dict

class SupportTriageAction(Action):
    """Action for the Support Triage environment - read, route, or reply."""

    action_type: Literal["read_ticket", "route_ticket", "reply_ticket", "done"] = Field(
        ..., description="The type of action to perform: 'read_ticket' a specific ticket, 'route_ticket' to a department, 'reply_ticket' directly to user, or 'done'."
    )
    ticket_id: Optional[int] = Field(None, description="The ID of the ticket to read, route, or reply to.")
    department: Optional[Literal["billing", "technical", "sales"]] = Field(
        None, description="The department to route the ticket to (required if action_type is 'route_ticket')."
    )
    reply_text: Optional[str] = Field(None, description="The text to reply with (required if action_type is 'reply_ticket').")


class SupportTriageObservation(Observation):
    """Observation from the Support Triage environment."""

    message: str = Field(..., description="System response resulting from the last action.")
    remaining_tickets: int = Field(..., description="Number of tickets remaining in the queue to be processed.")
    ticket_queue: List[Dict] = Field(..., description="List of tickets. Shows ID and subject preview if not explicitly read. Shows full text if read last turn.")
    knowledge_base: str = Field(..., description="The internal knowledge base for answering FAQ.")
