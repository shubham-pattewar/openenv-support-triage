"""Support ticket triage environment with deterministic graders and shaped rewards."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupportTriageAction, SupportTriageObservation
except ImportError:
    from models import SupportTriageAction, SupportTriageObservation


Department = Literal["billing", "technical", "sales"]
ResolutionType = Literal["route", "reply"]


@dataclass(frozen=True)
class TicketSpec:
    ticket_id: int
    subject: str
    preview: str
    full_text: str
    resolution_type: ResolutionType
    department: Optional[Department] = None
    reply_keywords: tuple[str, ...] = ()
    weight: float = 0.2
    ambiguity_penalty: float = 0.35
    recommended_read: bool = True
    notes: str = ""


TASK_OBJECTIVES: Dict[str, str] = {
    "easy": "Resolve three straightforward tickets using the correct team or FAQ answer.",
    "medium": "Process a mixed queue with one ambiguous routing case and one documentation reply.",
    "hard": "Handle a production-like queue with deceptive previews, escalation risk, and multiple FAQ replies.",
}

KNOWLEDGE_BASE = (
    "SUPPORT KNOWLEDGE BASE\n"
    "- Business hours: Monday-Friday, 9 AM-5 PM EST.\n"
    "- API docs: https://api.example.com/docs\n"
    "- Free trial: 14-day free trial, no credit card required.\n"
    "- Billing owns charges, refunds, invoices, accidental upgrades, and subscription plan disputes.\n"
    "- Technical owns bugs, crashes, authentication failures, integrations, and runtime errors.\n"
    "- Sales owns pricing, procurement, pilots, custom plans, and enterprise negotiations.\n"
)

TICKETS: Dict[int, TicketSpec] = {
    101: TicketSpec(
        ticket_id=101,
        subject="Duplicate charge on renewal",
        preview="Customer says they were charged twice after renewal.",
        full_text=(
            "Hi support, my card ending in 4242 was charged twice when our annual renewal ran "
            "this morning. Please refund the extra payment."
        ),
        resolution_type="route",
        department="billing",
        weight=0.30,
        ambiguity_penalty=0.10,
        recommended_read=False,
        notes="Straightforward billing case.",
    ),
    102: TicketSpec(
        ticket_id=102,
        subject="Upload flow returns 500",
        preview="Profile photo upload spins forever.",
        full_text=(
            "When I upload a profile picture, the request ends in a 500 error and the save button "
            "never completes on Chrome 124."
        ),
        resolution_type="route",
        department="technical",
        weight=0.30,
        ambiguity_penalty=0.10,
        recommended_read=False,
        notes="Straightforward technical defect.",
    ),
    103: TicketSpec(
        ticket_id=103,
        subject="Need enterprise pricing for 50 seats",
        preview="Procurement wants a quote for a larger rollout.",
        full_text=(
            "We are expanding to 50 seats next quarter and need an enterprise pricing quote with "
            "annual billing terms."
        ),
        resolution_type="route",
        department="sales",
        weight=0.25,
        ambiguity_penalty=0.10,
        recommended_read=False,
        notes="Straightforward sales lead.",
    ),
    104: TicketSpec(
        ticket_id=104,
        subject="What are your business hours?",
        preview="Customer asks when live support is available.",
        full_text="What are your business hours? We are based in New York and need to plan a call.",
        resolution_type="reply",
        reply_keywords=("9 am", "5 pm", "est", "monday", "friday"),
        weight=0.30,
        ambiguity_penalty=0.05,
        recommended_read=False,
        notes="FAQ reply using KB.",
    ),
    105: TicketSpec(
        ticket_id=105,
        subject="Where is the API documentation?",
        preview="Developer wants the public API docs.",
        full_text="Can you send the URL for your API documentation so our team can start integrating?",
        resolution_type="reply",
        reply_keywords=("api.example.com/docs", "api", "docs"),
        weight=0.30,
        ambiguity_penalty=0.05,
        recommended_read=False,
        notes="FAQ reply using KB.",
    ),
    106: TicketSpec(
        ticket_id=106,
        subject="Invoice page broken",
        preview="Subject mentions billing and invoice trouble.",
        full_text=(
            "My invoice total is fine, but when I click 'Pay now' the payment page crashes with a "
            "JavaScript exception. I cannot complete checkout because of the bug."
        ),
        resolution_type="route",
        department="technical",
        weight=0.34,
        ambiguity_penalty=0.40,
        recommended_read=True,
        notes="Ambiguous preview; actual owner is technical.",
    ),
    107: TicketSpec(
        ticket_id=107,
        subject="Urgent: unwanted upgrade charge",
        preview="Subject mentions technical trouble after a plan change.",
        full_text=(
            "Our workspace was automatically upgraded to Pro overnight and we were charged $149 "
            "without approval. Please reverse the charge and restore the prior plan."
        ),
        resolution_type="route",
        department="billing",
        weight=0.33,
        ambiguity_penalty=0.35,
        recommended_read=True,
        notes="Looks operational, but it is a billing dispute.",
    ),
    108: TicketSpec(
        ticket_id=108,
        subject="Mobile app crash after update",
        preview="iPhone app crashes on the messages tab.",
        full_text=(
            "After installing today's iOS update, the app crashes every time I open the messages tab "
            "on an iPhone 14 running iOS 18.1."
        ),
        resolution_type="route",
        department="technical",
        weight=0.33,
        ambiguity_penalty=0.10,
        recommended_read=False,
        notes="Straightforward technical issue.",
    ),
    109: TicketSpec(
        ticket_id=109,
        subject="Need a startup pilot plan",
        preview="Startup founder wants a custom pilot for 10 developers.",
        full_text=(
            "We are a startup evaluating your platform and would like to discuss a pilot plan for "
            "10 developers before committing to a yearly contract."
        ),
        resolution_type="route",
        department="sales",
        weight=0.33,
        ambiguity_penalty=0.10,
        recommended_read=False,
        notes="Straightforward sales inquiry.",
    ),
    110: TicketSpec(
        ticket_id=110,
        subject="Do you offer a free trial?",
        preview="Prospect asks whether a free trial exists.",
        full_text="Before I ask my manager to approve this tool, do you offer a free trial?",
        resolution_type="reply",
        reply_keywords=("14-day", "14 day", "free trial", "no credit card"),
        weight=0.34,
        ambiguity_penalty=0.05,
        recommended_read=False,
        notes="FAQ reply using KB.",
    ),
}

TASKS: Dict[str, List[int]] = {
    "easy": [101, 104, 102],
    "medium": [101, 106, 105, 103],
    "hard": [106, 107, 108, 110, 105, 109],
}


class SupportTriageEnvironment(Environment):
    """Real-world customer support triage environment."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    max_steps: int = 12

    def __init__(self) -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_task = "easy"
        self.queue: List[Dict[str, Any]] = []
        self.processed_ticket_ids: List[int] = []
        self.resolution_scores: Dict[int, float] = {}
        self.invalid_actions = 0
        self.loop_warnings = 0
        self.last_read_id: Optional[int] = None
        self.last_action_error: Optional[str] = None
        self.final_score = 0.0
        self.ticket_weights: Dict[int, float] = {}
        self.total_possible_score = 1.0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportTriageObservation:
        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        requested_task = str(
            kwargs.get("task") or kwargs.get("task_id") or os.getenv("SUPPORT_TRIAGE_TASK", "easy")
        ).lower()
        self.current_task = requested_task if requested_task in TASKS else "easy"

        self.queue = []
        self.processed_ticket_ids = []
        self.resolution_scores = {}
        self.invalid_actions = 0
        self.loop_warnings = 0
        self.last_read_id = None
        self.last_action_error = None
        self.final_score = 0.0

        task_ticket_ids = TASKS[self.current_task]
        self.ticket_weights = {ticket_id: TICKETS[ticket_id].weight for ticket_id in task_ticket_ids}
        self.total_possible_score = sum(self.ticket_weights.values())

        for ticket_id in task_ticket_ids:
            spec = copy.deepcopy(TICKETS[ticket_id])
            self.queue.append(
                {
                    "id": spec.ticket_id,
                    "subject": spec.subject,
                    "preview": spec.preview,
                    "full_text": spec.full_text,
                    "is_read": False,
                    "resolution_type": spec.resolution_type,
                    "recommended_read": spec.recommended_read,
                    "notes": spec.notes,
                }
            )

        return self._observation(
            message="Environment reset. Triage the queue using routing and FAQ replies.",
            reward=0.0,
            done=False,
        )

    def _queue_item(self, ticket: Dict[str, Any]) -> Dict[str, object]:
        item: Dict[str, object] = {
            "id": ticket["id"],
            "subject": ticket["subject"],
            "preview": ticket["preview"],
            "resolution_hint": ticket["resolution_type"],
        }
        if ticket["is_read"]:
            item["full_text"] = ticket["full_text"]
        return item

    def _normalized_score(self) -> float:
        resolved = sum(self.resolution_scores.values())
        loop_penalty = 0.03 * self.loop_warnings
        invalid_penalty = 0.05 * self.invalid_actions
        score = (resolved - loop_penalty - invalid_penalty) / self.total_possible_score
        return max(0.0, min(1.0, round(score, 6)))

    def _observation(self, message: str, reward: float, done: bool) -> SupportTriageObservation:
        score = self._normalized_score()
        if done:
            self.final_score = score

        return SupportTriageObservation(
            task_id=self.current_task,
            task_objective=TASK_OBJECTIVES[self.current_task],
            message=message,
            remaining_tickets=len(self.queue),
            ticket_queue=[self._queue_item(ticket) for ticket in self.queue],
            processed_ticket_ids=list(self.processed_ticket_ids),
            knowledge_base=KNOWLEDGE_BASE,
            last_action_error=self.last_action_error,
            grader_score=score if done else None,
            done=done,
            reward=max(0.0, min(1.0, reward)),
            metadata={
                "final_score": score if done else None,
                "invalid_actions": self.invalid_actions,
                "loop_warnings": self.loop_warnings,
                "step_budget": self.max_steps,
            },
        )

    def _get_ticket_index(self, ticket_id: Optional[int]) -> int:
        if ticket_id is None:
            return -1
        for index, ticket in enumerate(self.queue):
            if ticket["id"] == ticket_id:
                return index
        return -1

    def _invalid(self, message: str) -> tuple[float, str]:
        self.invalid_actions += 1
        self.last_action_error = message
        return 0.0, message

    def _score_reply(self, spec: TicketSpec, reply_text: str, read_first: bool) -> float:
        reply = reply_text.lower()
        hits = sum(1 for keyword in spec.reply_keywords if keyword in reply)
        if hits == 0:
            return 0.0
        quality = hits / max(1, len(spec.reply_keywords))
        base = spec.weight * (0.6 + 0.4 * quality)
        if spec.recommended_read and not read_first:
            base *= 1.0 - spec.ambiguity_penalty
        return min(spec.weight, round(base, 6))

    def step(self, action: SupportTriageAction) -> SupportTriageObservation:  # type: ignore[override]
        self._state.step_count += 1
        self.last_action_error = None

        if self._state.step_count > self.max_steps:
            self.loop_warnings += 1
            return self._observation(
                message="Step budget exhausted before all tickets were resolved.",
                reward=0.0,
                done=True,
            )

        score_before = self._normalized_score()

        if action.action_type == "done":
            if self.queue:
                self.loop_warnings += 1
                self.last_action_error = "done called before queue was empty"
                return self._observation(
                    message="You ended the episode before clearing the queue.",
                    reward=0.0,
                    done=True,
                )
            return self._observation(
                message="Queue already empty. Episode complete.",
                reward=0.0,
                done=True,
            )

        ticket_index = self._get_ticket_index(action.ticket_id)
        if ticket_index < 0:
            reward, message = self._invalid("ticket_id is missing or no longer in the queue")
            return self._observation(message=message, reward=reward, done=False)

        ticket = self.queue[ticket_index]
        spec = TICKETS[ticket["id"]]

        if action.action_type == "read_ticket":
            if ticket["is_read"]:
                self.loop_warnings += 1
                self.last_action_error = f"ticket {ticket['id']} was already read"
                return self._observation(
                    message=f"Ticket {ticket['id']} is already open. Re-reading wastes a step.",
                    reward=0.0,
                    done=False,
                )

            ticket["is_read"] = True
            self.last_read_id = ticket["id"]
            return self._observation(
                message=f"Opened ticket {ticket['id']}. Full details are now visible in ticket_queue.",
                reward=0.04 if spec.recommended_read else 0.02,
                done=False,
            )

        if action.action_type == "route_ticket":
            if action.department is None:
                reward, message = self._invalid("department is required for route_ticket")
                return self._observation(message=message, reward=reward, done=False)
            if spec.resolution_type != "route":
                reward, message = self._invalid(
                    f"ticket {ticket['id']} requires a customer reply, not routing"
                )
                return self._observation(message=message, reward=reward, done=False)

            read_first = bool(ticket["is_read"])
            if action.department == spec.department:
                resolved_score = spec.weight
                if spec.recommended_read and not read_first:
                    resolved_score *= 1.0 - spec.ambiguity_penalty
                self.resolution_scores[ticket["id"]] = round(resolved_score, 6)
                self.processed_ticket_ids.append(ticket["id"])
                self.queue.pop(ticket_index)
                message = f"Ticket {ticket['id']} routed to {action.department}."
            else:
                self.invalid_actions += 1
                self.last_action_error = (
                    f"ticket {ticket['id']} was routed to {action.department}, but that is incorrect"
                )
                self.resolution_scores[ticket["id"]] = 0.0
                self.processed_ticket_ids.append(ticket["id"])
                self.queue.pop(ticket_index)
                message = f"Ticket {ticket['id']} was misrouted."

        elif action.action_type == "reply_ticket":
            if action.reply_text is None:
                reward, message = self._invalid("reply_text is required for reply_ticket")
                return self._observation(message=message, reward=reward, done=False)
            if spec.resolution_type != "reply":
                reward, message = self._invalid(
                    f"ticket {ticket['id']} should be routed rather than replied to"
                )
                return self._observation(message=message, reward=reward, done=False)

            resolved_score = self._score_reply(spec, action.reply_text, bool(ticket["is_read"]))
            if resolved_score <= 0.0:
                self.invalid_actions += 1
                self.last_action_error = (
                    f"reply for ticket {ticket['id']} did not include the required knowledge-base facts"
                )
                message = f"Reply to ticket {ticket['id']} was insufficient."
            else:
                message = f"Ticket {ticket['id']} was answered from the knowledge base."

            self.resolution_scores[ticket["id"]] = resolved_score
            self.processed_ticket_ids.append(ticket["id"])
            self.queue.pop(ticket_index)

        else:
            reward, message = self._invalid(f"unsupported action_type: {action.action_type}")
            return self._observation(message=message, reward=reward, done=False)

        done = len(self.queue) == 0
        score_after = self._normalized_score()
        reward = max(0.0, min(1.0, round(score_after - score_before, 6)))
        if done:
            message = f"{message} Queue cleared."
        return self._observation(message=message, reward=reward, done=done)

    @property
    def state(self) -> State:
        return self._state
