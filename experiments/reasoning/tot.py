from langchain_openai import ChatOpenAI

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

from reasoning.tool import calculator, get_weather

import json
import re


# ============================================================
# LLM
# ============================================================

from pathlib import Path
from dotenv import load_dotenv
import os

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

llm = ChatOpenAI(
    model="OpenRouter/free",
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY")
)


# ============================================================
# STATE
# ============================================================

class Thought(TypedDict):
    path: str
    next_action: str
    score: float
    evaluation: str


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    user_goal: str
    reasoning: str
    next_action: str
    thoughts: list[Thought]
    current_step: int
    completed_steps: list[str]
    tool_results: list[str]


# ============================================================
# JSON HELPER
# ============================================================

def safe_json_load(text: str):

    if not text:
        raise ValueError("Model returned empty response.")

    text = text.strip()

    # Remove markdown fences
    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )

    # First try normal JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON array
    match = re.search(
        r"\[.*\]",
        text,
        re.DOTALL
    )

    if match:
        candidate = match.group(0)

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # Try extracting JSON object
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if match:
        candidate = match.group(0)

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON from model response:\n{text}"
    )


# ============================================================
# DEPTH 0
# Generate 3 ROOT thoughts
# ============================================================

def generate_thoughts(state: AgentState):

    prompt = f"""
You are performing Tree-of-Thoughts reasoning.

User goal:
{state["user_goal"]}

Completed actions:
{state["completed_steps"]}

Known tool results:
{state["tool_results"]}

Available tools:
- get_weather(city)
- calculator(expression)

IMPORTANT ACTION RULES:

Every "next_action" MUST be exactly one executable tool action.

Do NOT write natural-language actions.

Do NOT invent tools.

Do NOT use information that is not present in:
- the user goal
- completed actions
- known tool results

The next_action must be executable RIGHT NOW.

Generate exactly 3 DIFFERENT candidate next actions.

For each candidate:

- "path" describes the reasoning path beginning from the current state.
- "next_action" is ONLY the immediate action that should be executed now.
- The next_action must be executable using the current state and available tools.
- Do not include future actions inside next_action.
- Do not give the final answer.

Example:

{{
    "path": "Retrieve Berlin temperature → Retrieve London temperature → Calculate ratio",
    "next_action": "get_weather(city=\"Berlin\")"
}}

Return ONLY valid JSON.

Format:

[
    {{
        "path": "...",
        "next_action": "..."
    }},
    {{
        "path": "...",
        "next_action": "..."
    }},
    {{
        "path": "...",
        "next_action": "..."
    }}
]
"""

    response = llm.invoke(prompt)

    print(f"Raw response: {response.content}")

    try:
        thoughts = safe_json_load(response.content)
    except Exception as e:

        print("\n[WARNING] Root JSON failed:")
        print(repr(response.content))
        print(e)

        # Safe fallback
        thoughts = [
            {
                "path": "Determine the next required action",
                "next_action": "Determine the next required action"
            }
        ]

    # Normalize
    normalized = []

    for thought in thoughts[:3]:
        if (
            isinstance(thought, dict)
            and "path" in thought
            and "next_action" in thought
        ):
            normalized.append({
                "path": str(thought["path"]),
                "next_action": str(thought["next_action"]),
                "score": 0,
                "evaluation": ""
            })

    # Guarantee at least one thought
    if not normalized:
        normalized = [
            {
                "path": "Determine the next required action.",
                "next_action": "Determine the next required action.",
                "score": 0,
                "evaluation": ""
            }
        ]

    print("\n=== ROOT THOUGHTS ===")

    for i, thought in enumerate(normalized):
        print(f"{i + 1}. PATH: {thought['path']}")
        print(f"   NEXT: {thought['next_action']}")

    return normalized


# ============================================================
# DEPTH 1
# Expand ALL ROOTS in ONE LLM CALL
# ============================================================

def expand_all_thoughts(
    state: AgentState,
    root_thoughts: list[Thought]
):

    roots_text = "\n".join(
    f"""
    ROOT {i + 1}:
    Path: {thought['path']}
    Next action: {thought['next_action']}
    """
        for i, thought in enumerate(root_thoughts)
    )

    prompt = f"""
You are performing Tree-of-Thoughts expansion.

User goal:
{state["user_goal"]}

Completed actions:
{state["completed_steps"]}

Known tool results:
{state["tool_results"]}

Available tools:
- get_weather(city)
- calculator(expression)

IMPORTANT:

Every child MUST contain:

1. "path": the reasoning path after taking the parent action.
2. "next_action": ONLY the immediate executable action after the parent action.

Never describe an action in natural language.

Never invent tools.

Never invent tool results.

Never repeat an action that is already completed unless
the task explicitly requires a fresh retrieval.

Here are the root thoughts:

{roots_text}

For EACH root thought, generate exactly 2 possible child thoughts.

A child must logically continue from its parent.

IMPORTANT:
- Do not jump to unrelated actions.
- Do not invent tool results.
- Only use available tools.
- If a root already represents a terminal action, its children
  may represent the logical continuation after that action.
- Keep each child concise.

Return ONLY valid JSON in this exact structure:

[
  {{
    "parent": 1,
    "children": [
      {{"path": "...", "next_action": "..."}},
      {{"path": "...", "next_action": "..."}}
    ]
  }},
  {{
    "parent": 2,
    "children": [
      {{"path": "...", "next_action": "..."}},
      {{"path": "...", "next_action": "..."}}
    ]
  }},
  {{
    "parent": 3,
    "children": [
      {{"path": "...", "next_action": "..."}},
      {{"path": "...", "next_action": "..."}}
    ]
  }}
]
"""

    response = llm.invoke(prompt)

    try:
        expansions = safe_json_load(response.content)

    except Exception as e:

        print("\n[WARNING] Expansion JSON failed:")
        print(repr(response.content))
        print(e)

        # Fallback:
        expansions = []

        for i, root in enumerate(root_thoughts, start=1):

            expansions.append({
                "parent": i,
                "children": [
                    {
                        "path": root["path"],
                        "next_action": root["next_action"]
                    }
                ]
            })

    expanded = []

    for item in expansions:

        parent_idx = item.get("parent")

        if not isinstance(parent_idx, int):
            continue

        if parent_idx < 1 or parent_idx > len(root_thoughts):
            continue

        parent = root_thoughts[parent_idx - 1]

        children = item.get("children", [])

        for child in children[:2]:

            if not isinstance(child, dict):
                continue

            if "path" not in child:
                continue

            if "next_action" not in child:
                continue

            path = (
                parent["path"]
                + " → "
                + str(child["path"])
            )

            expanded.append({
                "path": path,
                "next_action": str(child["next_action"]),
                "score": 0,
                "evaluation": ""
            })

    # Safety fallback
    if not expanded:

        expanded = root_thoughts.copy()

    return expanded


# ============================================================
# EVALUATION
# ============================================================

def evaluate_thoughts(
    state: AgentState,
    thoughts: list[Thought]
):

    candidates = "\n".join(
        f"""
    Candidate {i + 1}:
    Path: {thought['path']}
    Next action: {thought['next_action']}
    """
        for i, thought in enumerate(thoughts)
    )

    prompt = f"""
You are evaluating Tree-of-Thoughts candidate paths.

User goal:
{state["user_goal"]}

Completed actions:
{state["completed_steps"]}

Candidate paths:

{candidates}

For EACH candidate, return:

- candidate_id
- score
- evaluation

IMPORTANT:

1. Do NOT rewrite the candidate path.
2. Do NOT generate a new path.
3. Do NOT modify next_action.
4. candidate_id must correspond to the candidate number.
5. Penalize actions that are already completed.
6. Penalize actions that cannot be executed from the current state.
7. Penalize redundant tool calls.
8. Penalize invented information.
9. Prefer the shortest valid path.
10. If the task is already complete, candidates requiring another
   tool call should receive a very low score.

Score from 1 to 10.

Return ONLY valid JSON:

[
    {{
        "candidate_id": 1,
        "score": 8,
        "evaluation": "..."
    }},
    {{
        "candidate_id": 2,
        "score": 5,
        "evaluation": "..."
    }}
]
"""

    response = llm.invoke(prompt)

    try:

        scored = safe_json_load(response.content)

    except Exception as e:

        print("\n[WARNING] Evaluation JSON failed:")
        print(repr(response.content))
        print(e)

        # If evaluator fails, don't crash the agent.
        # Give all candidates a neutral score.
        scored = [
                {
                    "candidate_id": i + 1,
                    "score": 5,
                    "evaluation": "fallback"
                }
                for i, thought in enumerate(thoughts)
            ]

    normalized = []

    for item in scored:
        if not isinstance(item, dict):
            continue

        try:
            candidate_id = int(item["candidate_id"])
            score = float(item.get("score", 5))
        except Exception:
            continue

        if not (1 <= candidate_id <= len(thoughts)):
            continue

        normalized.append({
            "path": thoughts[candidate_id - 1]["path"],
            "next_action": thoughts[candidate_id - 1]["next_action"],
            "score": score,
            "evaluation": str(
                item.get("evaluation", "")
            )
        })
    # Safety fallback
    if not normalized:

        normalized = [
            {
                "path": thought["path"],
                "next_action": thought["next_action"],
                "score": 5,
                "evaluation": "fallback"
            }
            for thought in thoughts
        ]

    print("\n=== SCORED THOUGHTS ===")

    for thought in normalized:
        print(
            f"{thought['score']}/10 -> "
            f"{thought['path']} "
            f"{thought['next_action']}"
        )

    return normalized


# ============================================================
# MAIN REASONING NODE
# ============================================================

def reason_node(state: AgentState):

    print("\n" + "=" * 60)
    print("TOT REASONING")
    print("=" * 60)

    print("\nCompleted:")
    print(state["completed_steps"])

    # --------------------------------------------------------
    # DEPTH 0
    # --------------------------------------------------------

    root_thoughts = generate_thoughts(state)

    # --------------------------------------------------------
    # DEPTH 1
    # --------------------------------------------------------

    expanded_thoughts = expand_all_thoughts(
        state,
        root_thoughts
    )

    print("\n=== EXPANDED TREE ===")

    for i, thought in enumerate(expanded_thoughts):
        print(
            f"{i + 1}. {thought['path']}"
        )

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    scored = evaluate_thoughts(
        state,
        expanded_thoughts
    )

    # --------------------------------------------------------
    # SELECT BEST PATH
    # --------------------------------------------------------

    best = max(
        scored,
        key=lambda x: x["score"]
    )

    print("\n=== BEST PATH ===")
    print(f"Path: {best['path']}")
    print(f"Next action: {best['next_action']}")
    print(f"Score: {best['score']}/10")
    print(f"Evaluation: {best['evaluation']}")

    return {
        "reasoning": best["path"],
        "next_action": best["next_action"],
        "thoughts": scored
    }


# ============================================================
# ACTION
# ============================================================

llm_with_tools = llm.bind_tools([
    calculator,
    get_weather
])

def record_tool_results(state: AgentState):

    new_results = list(state["tool_results"])

    for message in state["messages"]:
        if message.__class__.__name__ == "ToolMessage":

            result = str(message.content)

            if result not in new_results:
                new_results.append(result)

    return {
        "tool_results": new_results
    }

def act_node(state: AgentState):

    prompt = f"""
You are the action component of an AI agent.

User's ultimate goal:
{state["user_goal"]}

Current selected reasoning path:
{state["reasoning"]}

IMPORTANT:
Execute ONLY this immediate next action:

{state["next_action"]}

Completed actions:
{state["completed_steps"]}

Conversation:
{state["messages"]}

Execute ONLY the next appropriate action.

If a tool is required:
- call the appropriate tool
- use arguments implied by the reasoning

If all required information has been obtained:
- provide the final answer
- do not call unnecessary tools
"""

    print("\n=== ACTING ===")
    print(state["reasoning"])

    response = llm_with_tools.invoke(prompt)

    new_completed = list(
        state["completed_steps"]
    )

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


# ============================================================
# TOOLS
# ============================================================

tool_node = ToolNode([
    calculator,
    get_weather
])


# ============================================================
# ROUTING
# ============================================================

def should_continue(state: AgentState):

    if state["current_step"] >= 10:
        return "end"

    if state["messages"][-1].tool_calls:
        return "tools"

    return "end"


# ============================================================
# GRAPH
# ============================================================

graph = StateGraph(AgentState)

graph.add_node("reason", reason_node)
graph.add_node("agent", act_node)
graph.add_node("tools", tool_node)
graph.add_node("record_results", record_tool_results)

graph.add_edge(
    START,
    "reason"
)

graph.add_edge(
    "reason",
    "agent"
)

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

graph.add_edge(
    "tools",
    "record_results"
)

graph.add_edge(
    "record_results",
    "reason"
)

agent = graph.compile()


# ============================================================
# TEST
# ============================================================

user_question = (
    "What is the ratio of Berlin weather "
    "temperature to London?"
)

result = agent.invoke({

    "messages": [
        (
            "user",
            user_question
        )
    ],

    "user_goal": user_question,

    "reasoning": "",

    "thoughts": [],

    "next_action": "",

    "current_step": 0,

    "completed_steps": [],
    
    "tool_results": []
})


print("\n" + "=" * 60)
print("FINAL ANSWER")
print("=" * 60)

print(
    result["messages"][-1].content
)