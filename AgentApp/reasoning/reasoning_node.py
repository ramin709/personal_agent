from AgentApp.graph.state import AgentState
from AgentApp.initialize_llm import get_llm

def reason_node(state: AgentState):

    llm = get_llm()

    prompt = f"""
You are the reasoning component of an AI agent.

User goal:
{state["user_goal"]}

Current plan step:
{state["plan"][state["current_plan_step"]]}

Completed actions:
{state["completed_steps"]}

Conversation history:
{state["messages"]}

Determine which step of the existing plan should be executed next.

Do NOT create a new plan.

Identify:
1. What has already been completed.
2. What information is currently available.
3. Which planned step should be executed next.

Do not provide the final answer.

Return a concise reasoning summary and identify the immediate
next action from the existing plan.
"""

    result = llm.invoke(prompt)

    print("\n=== REASONING ===")
    print(result.content)

    return {
        "reasoning": result.content
    }