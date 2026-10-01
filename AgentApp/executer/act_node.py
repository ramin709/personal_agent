from AgentApp.graph.state import AgentState
from AgentApp.replanning.replanning_node import dependency_resolver
from AgentApp.initialize_llm import get_llm
from AgentTools.tool_registry import all_tools

def act_node(state: AgentState):
        
        needed_data = dependency_resolver(state, state["current_plan_step"])

        llm = get_llm()

        llm_with_tools = llm.bind_tools(all_tools.values())

        prompt = f"""
    You are the action component of an AI agent.

    User's ultimate goal:
    {state["user_goal"]}

    The plan for achieving this goal:
    {state['plan']}

    Current reasoning:
    {state["reasoning"]}

    Completed actions:
    {state["completed_steps"]}

    Conversation:
    {state["messages"]}

    needed data for this step:
    {needed_data}

    The plan is authoritative.

    Execute the next action required by the plan and identified
    by the reasoning.

    Do not invent a new action that is not part of the plan.

    If a tool is required:
    - call the appropriate tool
    - use the arguments implied by the reasoning

    If all required information has been obtained and no more
    tool calls are necessary, provide the final answer.
    """

        print("\n=== ACTING ===")
        print(state["reasoning"])

        response = llm_with_tools.invoke(prompt)

        # Only record actual tool actions.
        new_completed = list(state["completed_steps"])

        for tool_call in response.tool_calls:

            tool_name = tool_call["name"]
            args = tool_call["args"]

            new_completed.append(
                f"Called {tool_name} with {args}"
            )

        return {
            "messages": [response],
            "current_step": state["current_step"] + 1,
            "completed_steps": new_completed
        }
