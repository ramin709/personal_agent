from pydantic import BaseModel
from AgentApp.graph.state import AgentState
from AgentApp.replanning.replanning_node import dependency_resolver
from langchain_core.messages import ToolMessage
from AgentTools.tool_registry import all_tools

def execute_planned_tool(state: AgentState):

    current_step = state["plan"][state["current_plan_step"]]
    tool_name = current_step["tool"]

    try:
        args = dependency_resolver(
            state,
            state["current_plan_step"]
        )

        tool = all_tools[tool_name]
        result = tool.invoke(args)

        if isinstance(result, BaseModel):
            stored_result = result.model_dump()
        else:
            stored_result = result

        variables = dict(state["variables"])

        for output_name in current_step.get("output", []):
            variables[output_name] = stored_result

        new_completed = list(state["completed_steps"])
        new_completed.append(
            f"Called {tool_name} with {args}"
        )

        return {
            "messages": [
                ToolMessage(
                    content=str(result),
                    tool_call_id=f"planned_{current_step['step']}"
                )
            ],
            "current_step": state["current_step"] + 1,
            "completed_steps": new_completed,
            "variables": variables,
        }

    except Exception as e:

        error_message = (
            f"Tool '{tool_name}' failed.\n"
            f"Arguments: {args}\n"
            f"Error: {type(e).__name__}: {str(e)}"
        )

        return {
            "messages": [
                ToolMessage(
                    content=error_message,
                    tool_call_id=f"planned_{current_step['step']}"
                )
            ]
        }
