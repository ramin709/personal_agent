from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

    user_goal: str

    # CoT-style reasoning summary
    reasoning: str

    plan: list[dict]

    critic_comment: dict

    current_plan_step: int

    current_step: int

    completed_steps: list[str]

    variables: dict

    plan_origin: str

    plan_modified: bool

    retrieved_memory: list[dict]

    planning_history: list[dict]

    replanning_history: list[dict]