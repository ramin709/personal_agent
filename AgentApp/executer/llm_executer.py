from AgentApp.replanning.replanning_node import resolve_value
from AgentApp.graph.state import AgentState
from AgentApp.initialize_llm import get_llm

def execute_llm_step(state: AgentState):

    llm = get_llm()

    current_step = state["plan"][state["current_plan_step"]]

    args = resolve_value(
        current_step["tool_arguments"],
        state["variables"]
    )

    prompt = f"""
User goal:
{state["user_goal"]}

Task:
{current_step["plan_desc"]}

Input:
{args}

Perform this task and return the result.
"""

    result = llm.invoke(prompt)

    variables = dict(state["variables"])

    for output_name in current_step.get("output", []):
        variables[output_name] = result.content

    return {
        "reasoning": result.content,
        "variables": variables,
        "completed_steps": state["completed_steps"] + [
            f"Completed LLM step: {current_step['plan_desc']}"
        ],
        "current_step": state["current_step"] + 1
    }
