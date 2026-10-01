from AgentApp.graph.state import AgentState
from AgentApp.utils.json_parsing import safe_json_load
from AgentApp.initialize_llm import get_llm

def planning_node(state:AgentState):

    llm = get_llm()

    print(f"Found and inserted the following memories to the planner: {state['retrieved_memory']}")

    if not state["critic_comment"]:
        prompt = f"""
    You are the planning component of an AI agent.

    Your task is to create a complete execution plan BEFORE any
    actions are taken.

    User's ultimate goal:
    {state["user_goal"]}

    Retrieved memory:
    {state['retrieved_memory']}

    Available tools:
    - list_directory(path="...")
    - create_directory(path="...")
    - read_file(path="...")
    - write_file(path="...", content="...", overwrite=True/False)
    - search_repositories (query="...")
    - get_repository(owner= "ramin709", repo= "...")
    - list_repository_files(owner: "ramin709",repo= "...", path = "...",)
    - read_repository_file(owner: "ramin709", repo: "..." , path: "...",)

    Create a minimal, ordered sequence of actions required to
    achieve the goal. Plan steps must be the exact tool calls and arguments required to achieve that goal step.
    Each step must be atomic and containing only doing one job.
    If an argument depends on information that will only be available after a future step is executed,
    DO NOT invent the value. Use placeholders instead.
    This plan might have dependencies between steps, so make sure it considers what and how these dependencies are
    required between steps. The instructions is provided at the json examples. It should contains the dependency using 'depends_on' variable considering
    whether the specific step requires the output of the previous step(s) or not. So it doesn't depend solely on sequence of steps.
    Show the exact output name of each step in the 'output' variable, if a step requires the output of previous step/steps, depends_on and the output must be consistent.
    The name of output variables should be minimal but expressive like any variable name in programming.
    The plan also must specifies whether the step requires tool calling or not by 'type' variable that is either 'tool' or 'llm'

    DEPENDENCY REFERENCES:

    Use $ references whenever a step needs data produced by a previous step.

    Syntax:

    $variable
    $variable.field
    $variable.field[0]
    $variable.field[0].nested_field

    The root variable MUST exactly match one of the output variables
    declared by a dependency step.

    Examples:

    "$search_results.repositories[0].owner"
    "$search_results.repositories[0].name"
    "$file_list[0]"
    "$repository_info.default_branch"

    Never use:
    "TBD"
    "unknown"
    "<repository_name>"
    "{{repository_name}}"

    Rules:
    1. Plan the entire task from start to finish.
    2. Each step must be an executable action or a clear computation.
    3. Do not execute any tools.
    4. Do not provide the final answer.
    5. Do not include unnecessary steps.
    6. Do not assume information that is necessary for a decision but has not yet
    been obtained.

    Do not add inspection or discovery steps unless the result is actually
    required to determine a later action.
    7. Set "reasoning_required" according to whether additional reasoning is needed
    AT EXECUTION TIME.

    Set it to False when:
    - the tool is already determined,
    - all required arguments are already known,
    - no interpretation or decision is needed,
    - and the action can be executed directly.

    Set it to True only when execution requires:
    - interpreting information returned by a previous tool,
    - choosing between multiple possible actions,
    - deciding tool arguments that are not yet known,
    - resolving ambiguity,
    - or making a decision based on newly obtained information.

    The planning process itself does NOT count as execution-time reasoning.

    The length or complexity of tool arguments does NOT make reasoning necessary.
    A write_file operation with a fully specified path and content must use False.

    Remember to always put the output of the last step in the 'output' variable with the name final_outcome,
    Example: "output": ["final_outcome"] for the last step of the plan.


    Return a plan in only JSON with this format (The following is an example only):
    {{
        "plan" : [
            {{
                "step": 1,
                "plan_desc": "...",
                "tool": "...",
                "tool_arguments": {{"query": "value"}},
                "reasoning_required": False,
                "type": ""tool",
                "depends_on": [],
                "output": ["search_results"]

            }},

            {{
                "step": 2,
                "plan_desc": "...",
                "tool": "",
                "tool_arguments": {{"owner": "$search_results.repositories.0.owner",
            "repo": "$search_results.repositories.0.name"}},
                "reasoning_required": True,
                "type": "llm"",
                "depends_on": [1],
                "output": []
            }},
            .
            .
            .
        ]
    }}
    """

    else:
        prompt = f""" 
            You are the planning component of an AI agent.

        Your task is to create a complete execution plan BEFORE any
        actions are taken considering the critic reasoning provided.

        Your previous plan was:
        {state['plan']}

        And the critic comment is:
        {state["critic_comment"]}

        User's ultimate goal:
        {state["user_goal"]}

        Available tools:
        - list_directory(path="...")
        - create_directory(path="...")
        - read_file(path="...")
        - write_file(path="...", content="...", overwrite=True/False)
        - search_repositories (query="...")
        - get_repository(owner= "ramin709", repo= "...")
        - list_repository_files(owner: "ramin709",repo= "...", path = "...",)
        - read_repository_file(owner: "ramin709", repo: "..." , path: "...",)

        Create a minimal, ordered sequence of actions required to
        solve the issues critic mentioned. Plan steps must be the exact tool calls and arguments required to achieve that goal step.
        Each step must be atomic and containing only doing one job.
        If an argument depends on information that will only be available after a future step is executed,
        DO NOT invent the value. Use placeholders instead.
        This plan might have dependencies between steps, so make sure it considers what and how these dependencies are
        required between steps. The instructions is provided at the json examples. It should contains the dependency using 'depends_on' variable considering
        whether the specific step requires the output of the previous step(s) or not. So it doesn't depend solely on sequence of steps.
        Show the exact output name of each step in the 'output' variable, if a step requires the output of previous step/steps, depends_on and the output must be consistent.
        The name of output variables should be minimal but expressive like any variable name in programming.
        The plan also must specifies whether the step requires tool calling or not by 'type' variable that is either 'tool' or 'llm'

        DEPENDENCY REFERENCES:

        Use $ references whenever a step needs data produced by a previous step.

        Syntax:

        $variable
        $variable.field
        $variable.field[0]
        $variable.field[0].nested_field

        The root variable MUST exactly match one of the output variables
        declared by a dependency step.

        Examples:

        "$search_results.repositories[0].owner"
        "$search_results.repositories[0].name"
        "$file_list[0]"
        "$repository_info.default_branch"

        Never use:
        "TBD"
        "unknown"
        "<repository_name>"
        "{{repository_name}}"

        Rules:
        1. Plan the entire task from start to finish.
        2. Each step must be an executable action or a clear computation.
        3. Do not execute any tools.
        4. Do not provide the final answer.
        5. Do not include unnecessary steps.
        6. Do not assume information that is necessary for a decision but has not yet
        been obtained.

        Do not add inspection or discovery steps unless the result is actually
        required to determine a later action.
        7. Set "reasoning_required" according to whether additional reasoning is needed
        AT EXECUTION TIME.

        Set it to False when:
        - the tool is already determined,
        - all required arguments are already known,
        - no interpretation or decision is needed,
        - and the action can be executed directly.

        Set it to True only when execution requires:
        - interpreting information returned by a previous tool,
        - choosing between multiple possible actions,
        - deciding tool arguments that are not yet known,
        - resolving ambiguity,
        - or making a decision based on newly obtained information.

        The planning process itself does NOT count as execution-time reasoning.

        The length or complexity of tool arguments does NOT make reasoning necessary.
        A write_file operation with a fully specified path and content must use False.

        Remember to always put the output of the last step in the 'output' variable with the name final_outcome,
        Example: "output": ["final_outcome"] for the last step of the plan.

        Return a plan in only JSON with this format (The following is an example only):
        {{
            "plan" : [
                {{
                    "step": 1,
                    "plan_desc": "...",
                    "tool": "...",
                    "tool_arguments": {{"query": "value"}},
                    "reasoning_required": False,
                    "type": "tool",
                    "depends_on": [],
                    "output": ["search_results"]

                }},

                {{
                    "step": 2,
                    "plan_desc": "...",
                    "tool": "",
                    "tool_arguments": {{"owner": "$search_results.repositories.0.owner",
                "repo": "$search_results.items[0].name"}},
                    "reasoning_required": True,
                    "type": "llm",
                    "depends_on": [1],
                    "output": []
                }},
                .
                .
                .
            ]
        }}            

        """

        
    result = llm.invoke(prompt)

    print("\n=== Raw Planning ===")
    print(result.content)
    try:
        plan_steps = safe_json_load(result.content)
    except Exception as e:
        print("\n[WARNING] Root JSON failed:")
        print(repr(result.content))
        print(e)
    normalized = []
    for plan in plan_steps["plan"]:
        if (
            isinstance(plan, dict)
        ):
            normalized.append({
                "step": int(plan["step"]),
                "plan_desc": str(plan["plan_desc"]),
                "tool": str(plan["tool"]),
                "tool_arguments": dict(plan["tool_arguments"]),
                "reasoning_required": bool(plan["reasoning_required"]),
                "type": str(plan["type"]),
                "depends_on": list(plan.get("depends_on", [])),
                "output": list(plan.get("output", []))
            })
    return {
        "plan": normalized,
        "plan_origin": "planner"
    }

