from langchain_openai import ChatOpenAI

from typing import TypedDict, Annotated

from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

from reasoning.tool import calculator, get_weather
from eval.runner import run_benchmark, summarize
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


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    user_goal: str

    reasoning: str

    reflection: str

    current_step: int

    completed_steps: list[str]


def reason_node(state: AgentState):

    print("\n=== REASONING ===")

    prompt = f"""
You are the reasoning component of an AI agent.

User goal:
{state["user_goal"]}

Previous reasoning:
{state["reasoning"]}

Self-reflection:
{state["reflection"]}

Completed actions:
{state["completed_steps"]}

Conversation history:
{state["messages"]}

Use the previous reasoning and self-reflection to determine
the next useful action required to solve the goal.

If the reflection identifies an error or missing information,
correct it.

Reason through the problem step by step.

Do not provide the final answer.

Return a concise reasoning summary and the next action.
"""

    result = llm.invoke(prompt)

    print(result.content)

    return {
        "reasoning": result.content
    }


llm_with_tools = llm.bind_tools([
    calculator,
    get_weather
])

def reflection_node(state: AgentState):

    print("\n=== REFLECTION ===")

    prompt = f"""
You are the self-reflection component of an AI agent.

User goal:
{state["user_goal"]}

Current reasoning:
{state["reasoning"]}

Completed actions:
{state["completed_steps"]}

Conversation and tool results:
{state["messages"]}

Evaluate the agent's progress toward the user's goal.

Determine:

1. Whether the actions taken so far were correct.
2. Whether the obtained information is reliable and sufficient.
3. Whether any mistake or missing information exists.
4. What should happen next.

If everything is correct, explicitly say that no correction
is required.

Do not provide the final answer.

Return a concise reflection.
"""

    result = llm.invoke(prompt)

    print(result.content)

    return {
        "reflection": result.content
    }


def act_node(state: AgentState):

    prompt = f"""
You are the action component of an AI agent.

User's ultimate goal:
{state["user_goal"]}

Current reasoning:
{state["reasoning"]}

Completed actions:
{state["completed_steps"]}

Conversation:
{state["messages"]}

Based on the reasoning, execute the next appropriate action.

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


tool_node = ToolNode([
    calculator,
    get_weather
])


def should_continue(state: AgentState):

    if state["current_step"] >= 10:
        return "end"

    if state["messages"][-1].tool_calls:
        return "tools"

    return "end"


graph = StateGraph(AgentState)

graph.add_node("reason", reason_node)
graph.add_node("agent", act_node)
graph.add_node("tools", tool_node)
graph.add_node("reflect", reflection_node)

graph.add_edge(START, "reason")

graph.add_edge("reason", "agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

graph.add_edge("tools", "reflect")
graph.add_edge("reflect", "reason")

agent = graph.compile()


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

    "reflection": "",

    "current_step": 0,

    "completed_steps": []
})


print("\n=== FINAL ANSWER ===")
print(result["messages"][-1].content)


results = run_benchmark(agent=agent)

summary = summarize(results)

print("\n====================")
print("Self Reflection RESULTS")
print("====================")

for key, value in summary.items():
    print(f"{key}: {value}")