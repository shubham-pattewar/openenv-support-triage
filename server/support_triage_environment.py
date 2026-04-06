import os
import copy
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SupportTriageAction, SupportTriageObservation
except ImportError:
    from models import SupportTriageAction, SupportTriageObservation


TICKETS_DB = {
    101: {"department": "billing", "text": "I was charged twice on my credit card last week! Please help. My account is john@example.com.", "type": "route"},
    102: {"department": "technical", "text": "I'm getting a 500 internal server error when I try to upload my profile picture. The button just spins.", "type": "route"},
    103: {"department": "sales", "text": "Can I get a volume discount for 50 users on the enterprise plan? We are comparing you with your competitor.", "type": "route"},
    104: {"text": "What are your business hours?", "type": "reply", "valid_replies": ["9", "5", "est", "9am", "5pm"]},
    105: {"text": "Where can I find the API documentation?", "type": "reply", "valid_replies": ["api.example.com", "api"]},
    106: {"department": "billing", "text": "I need a copy of my last invoice for my accounting department.", "type": "route"},
    107: {"department": "technical", "text": "The mobile app keeps crashing on my iPhone 14 whenever I open the messages tab.", "type": "route"},
}

TASKS = {
    "easy": [101],
    "medium": [101, 102, 103],
    "hard": [101, 107, 103, 104, 105],
}

KB_TEXT = "KNOWLEDGE BASE: 1. Business hours are 9 AM to 5 PM EST, Monday-Friday. 2. API documentation is located at api.example.com."

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

    def reset(self) -> SupportTriageObservation:
        """Reset the environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        
        # Determine task from environment variable, default to 'easy'
        self.current_task = os.getenv("SUPPORT_TRIAGE_TASK", "easy").lower()
        if self.current_task not in TASKS:
            self.current_task = "easy"
            
        ticket_ids = TASKS[self.current_task]
        self.queue = copy.deepcopy([
            {"id": tid, "subject": TICKETS_DB[tid]["text"][:20] + "...", "full_text": TICKETS_DB[tid]["text"]}
            for tid in ticket_ids
        ])
        
        self.total_tickets = len(self.queue)
        self.last_read_id = None

        return self._generate_observation("Environment reset. Ready to triage tickets.", 0.0, False)

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
        
        # Calculate step reward value for a single successful action
        val_per_ticket = 1.0 / self.total_tickets if self.total_tickets > 0 else 0.0

        if action.action_type == "read_ticket":
            if action.ticket_id is not None:
                idx, t = self._get_ticket_from_queue(action.ticket_id)
                if t:
                    self.last_read_id = action.ticket_id
                    msg = f"Read ticket {action.ticket_id}."
                else:
                    msg = f"Ticket {action.ticket_id} not found."
            else:
                msg = "ticket_id is required."

        elif action.action_type == "route_ticket":
            if action.ticket_id is not None and action.department is not None:
                idx, t = self._get_ticket_from_queue(action.ticket_id)
                if t:
                    correct_dept = TICKETS_DB[action.ticket_id].get("department")
                    correct_type = TICKETS_DB[action.ticket_id].get("type")
                    
                    if correct_type == "route" and action.department == correct_dept:
                        msg = f"Successfully routed ticket {action.ticket_id} to {action.department}!"
                        reward = val_per_ticket
                    else:
                        msg = f"Incorrect routing for ticket {action.ticket_id}!"
                        
                    self.queue.pop(idx) # consumed
                    if self.last_read_id == action.ticket_id:
                        self.last_read_id = None
                else:
                    msg = f"Ticket {action.ticket_id} not found."
            else:
                msg = "ticket_id and department are required."
                
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
                    
                    self.queue.pop(idx) # consumed
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
