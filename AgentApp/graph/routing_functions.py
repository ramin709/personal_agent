from AgentApp.graph.state import AgentState

def update_progress(state):
    return {
        "current_plan_step":
        state["current_plan_step"] + 1
    }

def should_continue(state: AgentState):

    if state["current_step"] >= 10:
        return "end"

    if state["messages"][-1].tool_calls:
        return "tools"

    return "memory_store"

def isPlanFinished(state: AgentState):

    if state["current_plan_step"] >= len(state["plan"]):
        return "end"
    else:
        return "reason"
    
def should_replan(state):

    last_message = state["messages"][-1]

    print(f"Last message is: {last_message}")

    if "ERROR" in last_message.content:
        return "replanning"

    return "reason"

def route_plan_step(state: AgentState):

    if state["current_plan_step"] >= len(state["plan"]):
        return "memory_store"
    
    if not state["critic_comment"]["valid"]:
        
        if state["plan_origin"] == "planner":
            return "plan"
        elif state["plan_origin"] == "replanner":
            return "replanning"
        else:
            raise RuntimeError("Invalid origin was detected")

    current_step = state["plan"][state["current_plan_step"]]

    step_type = current_step["type"]
    reasoning_required = current_step["reasoning_required"]

    if step_type == "llm":
        return "llm"

    if step_type == "tool":
        if reasoning_required:
            return "reason"

        return "execute"

    raise ValueError(
        f"Unknown step type: {step_type}"
    )

def route_after_replanning(state):

    if state["plan_modified"]:
        return "critic"

    return route_plan_step(state)