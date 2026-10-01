from AgentApp.graph.state import AgentState
from AgentApp.utils.json_parsing import safe_json_load
from AgentApp.initialize_llm import get_llm

def replanning(state: AgentState):

    llm = get_llm()

    print(f"last message is: \n {state['messages'][-1]}")

    prompt = f""" 

You are the replanning node of an AI Agent.

Your job is to repair the CURRENT PLAN after the latest tool/action result reveals a failure, unexpected result, or new information.

The goal is NOT merely to repair the failed step.

You must ensure that the ENTIRE REMAINING PLAN is compatible with the information currently available after the failure.

---

## CURRENT PLAN AND EXECUTION STATE

User goal:

{state["user_goal"]}

Current plan:

{state["plan"]}

Current planning step:

{state["current_plan_step"]}

Completed actions:

{state["completed_steps"]}

Last action result:

{state["messages"][-1]}

Available working-memory variables:

{state["variables"]}

Previous replanning history:

{state["replanning_history"]}

---

## AVAILABLE TOOLS

* list_directory(path="...")
* create_directory(path="...")
* read_file(path="...")
* write_file(path="...", content="...", overwrite=True/False)
* search_repositories(query="...")
* get_repository(owner="...", repo="...")
* list_repository_files(owner="...", repo="...", path="...")
* read_repository_file(owner="...", repo="...", path="...")

---

# REPLANNING PROCEDURE

Before generating any patch, perform the following reasoning internally:

### 1. Identify the failure

Determine exactly what failed or what new information was discovered from the latest action result.

Do not assume anything that is not supported by:

* the user goal
* completed actions
* the last action result
* available variables
* the current plan
* known tool contracts

### 2. Inspect the remaining plan

Inspect the CURRENT PLAN from the current step onward.

Determine which future steps:

* depend on the failed step
* depend on its output
* assume that the failed action succeeded
* assume the existence of a file, directory, repository, value, or other resource that has now been shown to be invalid
* require information that is no longer available after the failure

A future step may need modification even if it was not the step that originally failed.

### 3. Repair the dependency chain

Your patches must make the remaining plan executable again.

If a future step depends on an invalidated assumption, modify that future step as well.

Do NOT leave a future step that requires information which the repaired plan will no longer produce.

For example:

If:

Step 3 = read main.py
Step 4 = analyze main.py

and Step 3 fails because main.py does not exist,

it is NOT sufficient to replace Step 3 with:

Step 3 = read README.md

because Step 4 still requires main.py content.

Instead, the remaining plan must first discover the actual file location, then read the discovered file, then analyze that file.

### 4. Discover instead of guessing

If the required resource or path is unknown, DO NOT guess it.

Add a discovery step using an appropriate tool.

For example, if a requested GitHub file does not exist and its replacement location is unknown:

1. list_repository_files(..., path="")
2. use an LLM reasoning step to determine the relevant file if necessary
3. read the selected file
4. continue with the dependent analysis

Never invent directory names, filenames, repository names, owners, or paths.

### 5. Preserve valid work

Never modify completed steps.

Do not repeat successful completed actions unless the failure makes their results invalid.

Do not modify a future step if it remains valid and compatible with the repaired dependency chain.

### 6. Preserve dynamic dependencies

When information already exists in working memory, reference it dynamically.

For example:

"$search_results.repositories[0].owner"

and:

"$search_results.repositories[0].name"

Do NOT replace dynamic references with hardcoded values.

### 7. Check the resulting plan before returning patches

Mentally apply your proposed patches to the CURRENT PLAN.

Then verify:

* Every step has a valid tool or is a valid LLM step.
* Every tool argument is available or dynamically referenced.
* No step depends on itself.
* No step depends on a future step.
* Every required output is produced before it is consumed.
* Every future step has access to the information it requires.
* No future step relies on a failed or invalidated assumption.
* No unnecessary step has been added.
* The remaining plan still leads toward the user goal.

Your patches must produce a coherent plan AFTER the executor applies them.

---

# PATCH RULES

You are NOT allowed to regenerate the entire plan.

You may only return modifications to the CURRENT PLAN.

Allowed actions:

1. replace_step
2. insert_after

You may modify:

* the current failed step
* any future step affected by the failure

You MUST NOT modify completed steps.

The "step" field identifies the target step in the CURRENT PLAN.

If you insert a step, DO NOT renumber existing steps.

The executor will renumber the plan after applying all patches.

If a future step becomes invalid because of your repair, replace that future step rather than leaving the invalid step unchanged.

---

# STEP RULES

1. Plan only the necessary steps.

2. Each step must be atomic and perform exactly one job.

3. Each step must be executable.

4. Tool steps must contain an exact tool and valid arguments.

5. LLM steps must have:

   * `"tool": ""`
   * `"type": "llm"`
   * `"reasoning_required": true`

6. Use `"reasoning_required": false` when the step can be completed by a deterministic tool call.

7. Use `"reasoning_required": true` when the step requires interpretation, selection, analysis, or decision-making.

8. Do not execute tools.

9. Do not provide the final answer.

10. Do not assume information that has not been obtained.

11. Do not use a tool from the wrong domain.
    For example, use `list_repository_files` for GitHub repositories, not `list_directory`.

12. `depends_on` represents DATA DEPENDENCY, not merely execution order.

A step should depend on another step only when it requires information produced by that step.

13. Every `$variable` reference must refer to an output that is actually available in working memory.

14. Use bracket indexing for lists:
    `$search_results.repositories[0].owner`

Do NOT use:
`$search_results.repositories.0.owner`

15. If a required value is unknown, add a discovery/reasoning step rather than guessing.

16. Never create a self-dependency.

17. Never create a dependency on a future step.

18. Never leave a future step that requires output which the repaired plan does not produce.

---

# OUTPUT FORMAT

If replanning is required, return ONLY:

{{
"replanning_required": true,
    "steps_modifications": [
        {{
            "action": "replace_step" | "insert_after",
            "step": 1,
            "plan_desc": "...",
            "tool": "...",
            "tool_arguments": {{}},
            "reasoning_required": false,
            "type": "tool",
            "depends_on": [],
            "output": []
        }}
    ]
}}

Otherwise return ONLY:

{{
"replanning_required": false
}}
 """
    result = llm.invoke(prompt)

    print("\n=== Raw Replanning ===")
    print(result.content)

    try:
        replanning = safe_json_load(result.content)
    except Exception as e:
        print("\n[WARNING] Root JSON failed:")
        print(repr(result.content))
        print(e)

    
    normalized = []
    if replanning["replanning_required"]:
        for step in replanning["steps_modifications"]:
            if (
                isinstance(step, dict)
            ):
                normalized.append({
                    "action": str(step["action"]),
                    "step": int(step["step"]),
                    "plan_desc": str(step["plan_desc"]),
                    "tool": str(step["tool"]),
                    "tool_arguments": dict(step["tool_arguments"]),
                    "reasoning_required": bool(step["reasoning_required"]),
                    "type": str(step["type"]),
                    "depends_on": list(step.get("depends_on", [])),
                    "output": list(step.get("output", []))
                })

        plan = replan_patch_resolver(state["plan"], normalized)
        return {
            "plan": plan,
            "plan_origin": "replanner",
            "plan_modified": True
        }
    else:
        return {
            "current_plan_step": state["current_plan_step"] + 1,
            "plan_modified": False,
            "critic_comment": {"valid": True, "issues": []}
        }
    
def replan_patch_resolver(plan, patches):

    plan_copy = [dict(step) for step in plan]

    for patch in patches:

        action = patch["action"]
        target_step = patch["step"]

        if action == "replace_step":

            idx = next(
                i for i, item in enumerate(plan_copy)
                if item["step"] == target_step
            )

            new_step = {
                "step": target_step,
                "plan_desc": patch["plan_desc"],
                "tool": patch["tool"],
                "tool_arguments": patch["tool_arguments"],
                "reasoning_required": patch["reasoning_required"],
                "type": patch["type"],
                "depends_on": patch.get("depends_on", []),
                "output": patch.get("output", [])
            }

            plan_copy[idx] = new_step

        elif action == "insert_after":

            idx = next(
                i for i, item in enumerate(plan_copy)
                if item["step"] == target_step
            )

            new_step = {
                "step": None,
                "plan_desc": patch["plan_desc"],
                "tool": patch["tool"],
                "tool_arguments": patch["tool_arguments"],
                "reasoning_required": patch["reasoning_required"],
                "type": patch["type"],
                "depends_on": patch.get("depends_on", []),
                "output": patch.get("output", [])
            }

            plan_copy.insert(idx + 1, new_step)

    # Renumber
    old_to_new = {}

    for new_number, step in enumerate(plan_copy, start=1):
        old_number = step["step"]

        if old_number is not None:
            old_to_new[old_number] = new_number

        step["step"] = new_number

    # Remap dependencies
    for step in plan_copy:
        step["depends_on"] = [
            old_to_new.get(dep, dep)
            for dep in step["depends_on"]
        ]

    return plan_copy

    
import re

def parse_reference(expression):
    expression = expression[1:]  # remove $

    parts = re.split(r'(\[\d+\])|\.', expression)

    tokens = []

    for part in parts:
        if not part:
            continue

        if part.startswith("[") and part.endswith("]"):
            tokens.append(int(part[1:-1]))
        else:
            tokens.append(part)

    return tokens

def resolve_value(value, variables):

    # $reference
    if isinstance(value, str) and value.startswith("$"):

        path = parse_reference(value)

        root = path[0]

        if root not in variables:
            raise KeyError(
                f"Variable '{root}' not found in working memory."
            )

        current = variables[root]

        for key in path[1:]:

            if isinstance(current, dict):
                current = current[key]

            elif isinstance(current, list):
                if not isinstance(key, int):
                    raise TypeError(
                        f"List indices must be integers, got '{key}'."
                    )

                current = current[key]

            else:
                raise TypeError(
                    f"Cannot access '{key}' from {type(current).__name__}."
                )

        return current

    # nested dict
    if isinstance(value, dict):
        return {
            key: resolve_value(val, variables)
            for key, val in value.items()
        }

    # nested list
    if isinstance(value, list):
        return [
            resolve_value(item, variables)
            for item in value
        ]

    return value
    
def dependency_resolver(state: AgentState, step_index: int):

    current_step = state["plan"][step_index]

    args = current_step["tool_arguments"]
    variables = state["variables"]

    resolved_args = resolve_value(args, variables)

    print("\n=== DEPENDENCY RESOLUTION ===")
    print(f"Original args: {args}")
    print(f"Resolved args: {resolved_args}")

    return resolved_args
