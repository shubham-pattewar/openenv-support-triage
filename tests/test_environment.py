from models import SupportTriageAction
from server.support_triage_environment import SupportTriageEnvironment


def test_easy_task_reaches_perfect_score():
    env = SupportTriageEnvironment()
    obs = env.reset(task="easy")
    assert obs.remaining_tickets == 3

    obs = env.step(SupportTriageAction(action_type="route_ticket", ticket_id=101, department="billing"))
    assert obs.reward > 0.0

    obs = env.step(SupportTriageAction(action_type="reply_ticket", ticket_id=104, reply_text="Support is available Monday-Friday, 9 AM to 5 PM EST."))
    assert obs.reward > 0.0

    obs = env.step(SupportTriageAction(action_type="route_ticket", ticket_id=102, department="technical"))
    assert obs.done is True
    assert obs.grader_score == 1.0


def test_hard_task_penalizes_unread_ambiguous_ticket():
    env = SupportTriageEnvironment()
    env.reset(task="hard")

    obs = env.step(SupportTriageAction(action_type="route_ticket", ticket_id=106, department="technical"))
    assert obs.done is False

    obs = env.step(SupportTriageAction(action_type="done"))
    assert obs.done is True
    assert 0.0 <= (obs.grader_score or 0.0) < 1.0
