import asyncio
import os
import sys

# Ensure current directory is in path
sys.path.insert(0, os.getcwd())

from client import SupportTriageEnv
from models import SupportTriageAction

async def demonstrate():
    print("Starting Project Logic Demonstration...")
    
    # Connect to the local server (assuming it's running)
    env = SupportTriageEnv(base_url="http://localhost:8000")
    
    try:
        # 1. Reset for Easy Task (1 Billing Ticket)
        os.environ["SUPPORT_TRIAGE_TASK"] = "easy"
        step_res_reset = await env.reset()
        obs = step_res_reset.observation
        print(f"Task Initialized: {obs.message}")
        print(f"Tickets in Queue: {obs.remaining_tickets}")
        
        # 2. Read the ticket
        ticket_id = obs.ticket_queue[0]["id"]
        print(f"\nAction: Reading ticket {ticket_id}...")
        action = SupportTriageAction(action_type="read_ticket", ticket_id=ticket_id)
        step_res = await env.step(action)
        print(f"Observation: {step_res.observation.message}")
        
        # 3. Route to Billing
        print(f"\nAction: Routing ticket {ticket_id} to 'billing'...")
        action = SupportTriageAction(action_type="route_ticket", ticket_id=ticket_id, department="billing")
        step_res = await env.step(action)
        print(f"Observation: {step_res.observation.message}")
        print(f"Reward Received: {step_res.reward}")
        
        if step_res.reward == 1.0:
            print("\n[SUCCESS]: Project logic is working perfectly! (1.0 reward issued)")
        else:
            print(f"\n[FAILED]: Received reward {step_res.reward} instead of 1.0")
            
    except Exception as e:
        print(f"\n[ERROR] during demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(demonstrate())
