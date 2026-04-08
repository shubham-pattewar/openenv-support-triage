import os
import copy
from uuid import uuid4
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupportTriageAction, SupportTriageObservation
except ImportError:
    from models import SupportTriageAction, SupportTriageObservation


TICKETS_DB = {
    # Clear-cut routing tickets
    101: {"department": "billing",   "text": "I was charged twice on my credit card last week! Please help. My account is john@example.com.", "type": "route"},
    102: {"department": "technical", "text": "I'm getting a 500 internal server error when I try to upload my profile picture. The button just spins forever.", "type": "route"},
    103: {"department": "sales",     "text": "Can I get a volume discount for 50 users on the enterprise plan? We are comparing you with competitor pricing.", "type": "route"},
    # FAQ / reply tickets
    104: {"text": "What are your business hours?",               "type": "reply", "valid_replies": ["9", "5", "est", "9am", "5pm", "monday", "friday"]},
    105: {"text": "Where can I find the API documentation?",     "type": "reply", "valid_replies": ["api.example.com", "api"]},
    # TRAP tickets — topic label in subject is misleading; correct department differs from naive reading
    106: {
        "department": "technical",
        "text": (
            "BILLING DEPARTMENT — Please help. My invoice shows the correct amount but "
            "after I click 'Pay Now', the payment page crashes with a JavaScript error. "
            "I cannot complete the payment due to this bug."
        ),
        "type": "route",
        "trap": True,  # Mentions billing but is actually a technical bug
    },
    107: {
        "department": "billing",
        "text": (
            "TECH SUPPORT — Hi, my account was automatically upgraded to the Pro plan yesterday "
            "without my consent and I was charged $149. I did not request this upgrade. "
            "Please reverse the charge immediately."
        ),
        "type": "route",
        "trap": True,  # Mentions tech support but is a billing dispute
    },
    108: {"department": "technical", "text": "The mobile app keeps crashing on my iPhone 14 whenever I open the messages tab after the latest update.", "type": "route"},
    109: {"department": "sales",     "text": "We are a startup and would like to discuss a custom pilot program for 10 developers. Who should we talk to?", "type": "route"},
    110: {"text": "Do you offer a free trial?", "type": "reply", "valid_replies": ["14-day", "14 day", "free trial", "trial"]},
}

SUPPORT_TRIAGE_TASKS = {
    "easy":   [101, 104],
    "medium": [101, 102, 103],
    "hard":   [101, 106, 107, 104, 108, 110],
}

KB_TEXT = (
    "KNOWLEDGE BASE: "
    "1. Business hours are 9 AM to 5 PM EST, Monday-Friday. "
    "2. API documentation is located at api.example.com. "
    "3. We offer a 14-day free trial, no credit card required. "
    "4. Route payment/charge/invoice disputes to billing. "
    "5. Route app crashes, errors, and technical bugs to technical. "
    "6. Route discount, pricing, and sales inquiries to sales."
)

class SupportTriageEnvironment(Environment):
    """
    Customer Support Triage environment.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the support_triage environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.queue = []
        self.total_tickets = 0
        self.last_read_id = None
        self.current_task = "easy"
        self.read_count = 0 

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportTriageObservation:
        """Reset the environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        
        # Determine task from kwargs or environment variable, default to 'easy'
        self.current_task = "easy"
        if "task" in kwargs:
            self.current_task = str(kwargs["task"]).lower()
        elif "task_id" in kwargs:
            self.current_task = str(kwargs["task_id"]).lower()
        else:
            self.current_task = os.getenv("SUPPORT_TRIAGE_TASK", "easy").lower()
        if self.current_task not in SUPPORT_TRIAGE_TASKS:
            self.current_task = "easy"
            
        ticket_ids = SUPPORT_TRIAGE_TASKS[self.current_task]
        self.queue = []
        for tid in ticket_ids:
            ticket = copy.deepcopy(TICKETS_DB[tid])
            self.queue.append({
                "id": tid,
                "subject": ticket["text"][:30] + "...",
                "full_text": ticket["text"]
            })
        
        self.total_tickets = len(self.queue)
        self.last_read_id = None
        self.read_count = 0

        return self._generate_observation("Environment reset. Ready to triage tickets.", 0.01, False)

    def _generate_observation(self, message: str, reward: float, done: bool) -> SupportTriageObservation:
        ticket_queue_view = []
        for ticket in self.queue:
            if self.last_read_id == ticket["id"]:
                ticket_queue_view.append({"id": ticket["id"], "text": ticket["full_text"]})
            else:
                ticket_queue_view.append({"id": ticket["id"], "text": ticket["subject"]})
                
        return SupportTriageObservation(
            message=message,
            remaining_tickets=len(self.queue),
            ticket_queue=ticket_queue_view,
            knowledge_base=KB_TEXT,
            done=done,
            reward=reward,
        )

    def _get_ticket_from_queue(self, qid):
        for i, t in enumerate(self.queue):
            if t["id"] == qid:
                return i, t
        return -1, None

    def step(self, action: SupportTriageAction) -> SupportTriageObservation:  # type: ignore[override]
        """Execute a step in the environment."""
        self._state.step_count += 1
        
        if len(self.queue) == 0 or action.action_type == "done":
            return self._generate_observation("All tickets processed or agent called done.", 0.0, True)

        reward = 0.0
        msg = f"Action {action.action_type} recognized."
        done = False
        
        val_per_ticket = 0.98 / self.total_tickets if self.total_tickets > 0 else 0.0

        if action.action_type == "read_ticket":
            if action.ticket_id is not None:
                idx, t = self._get_ticket_from_queue(action.ticket_id)
                if t:
                    self.last_read_id = action.ticket_id
                    self.read_count += 1
                    msg = f"Read ticket {action.ticket_id}. Full text now visible in ticket_queue."
                else:
                    msg = f"Ticket {action.ticket_id} not found."
            else:
                msg = "ticket_id is required for read_ticket."

        elif action.action_type == "route_ticket":
            if action.ticket_id is not None and action.department is not None:
                idx, t = self._get_ticket_from_queue(action.ticket_id)
                if t:
                    correct_dept = TICKETS_DB[action.ticket_id].get("department")
                    correct_type = TICKETS_DB[action.ticket_id].get("type")
                    is_trap = TICKETS_DB[action.ticket_id].get("trap", False)
                    was_read = self.last_read_id == action.ticket_id
                    
                    if correct_type == "route" and action.department == correct_dept:
                        if is_trap:
                            if was_read:
                                reward = val_per_ticket
                                msg = f"Excellent! Correctly identified and routed trap ticket {action.ticket_id} to {action.department} after reading!"
                            else:
                                reward = val_per_ticket * 0.5
                                msg = f"You correctly routed trap ticket {action.ticket_id}, but you didn't read it first. Partial reward given."
                        else:
                            reward = val_per_ticket
                            msg = f"Successfully routed ticket {action.ticket_id} to {action.department}!"
                    else:
                        msg = f"Incorrect routing for ticket {action.ticket_id}. No reward given."
                        
                    self.queue.pop(idx)
                    if self.last_read_id == action.ticket_id:
                        self.last_read_id = None
                else:
                    msg = f"Ticket {action.ticket_id} not found in queue."
            else:
                msg = "ticket_id and department are required for route_ticket."
                
        elif action.action_type == "reply_ticket":
            if action.ticket_id is not None and action.reply_text is not None:
                idx, t = self._get_ticket_from_queue(action.ticket_id)
                if t:
                    correct_type = TICKETS_DB[action.ticket_id].get("type")
                    
                    if correct_type == "reply":
                        valid_answers = TICKETS_DB[action.ticket_id].get("valid_replies", [])
                        replied_correctly = any(ans in action.reply_text.lower() for ans in valid_answers)
                        
                        if replied_correctly:
                            msg = f"Successfully replied to ticket {action.ticket_id}!"
                            reward = val_per_ticket
                        else:
                            msg = f"Incorrect reply for ticket {action.ticket_id}."
                    else:
                        msg = f"Ticket {action.ticket_id} should have been routed, not replied to!"
                    
                    self.queue.pop(idx)
                    if self.last_read_id == action.ticket_id:
                        self.last_read_id = None
                else:
                    msg = f"Ticket {action.ticket_id} not found."
            else:
                msg = "ticket_id and reply_text are required."
                
        if len(self.queue) == 0:
            msg += " All tickets cleared!"
            done = True

        return self._generate_observation(msg, reward, done)

    @property
    def state(self) -> State:
        return self._state
