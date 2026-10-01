from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from AgentTools.tool_registry import all_tools
from AgentApp.planning.planning_node import planning_node
from AgentApp.replanning.replanning_node import replanning
from AgentApp.reasoning.reasoning_node import reason_node
from AgentApp.executer.act_node import act_node
from AgentApp.executer.llm_executer import execute_llm_step
from AgentApp.executer.tool_executer import execute_planned_tool
from AgentApp.memory.memory_retrieval_node import memory_node
from AgentApp.memory.memory_store_node import memory_store_node
from AgentApp.critic.plan_critic_node import plan_critic_node
from .routing_functions import route_after_replanning,route_plan_step,should_replan,should_continue,update_progress
from .state import AgentState

def build_graph():

    tool_node = ToolNode(list(all_tools.values()))


    graph = StateGraph(AgentState)

    graph.add_node("reason", reason_node)
    graph.add_node("agent", act_node)
    graph.add_node("plan", planning_node)
    graph.add_node("tools", tool_node)
    graph.add_node("replanning", replanning)
    graph.add_node("update_progress", update_progress)
    graph.add_node("execute", execute_planned_tool)
    graph.add_node("llm", execute_llm_step)
    graph.add_node("critic", plan_critic_node)
    graph.add_node("memory", memory_node)
    graph.add_node("memory_store", memory_store_node)

    graph.add_edge(START, "memory")

    graph.add_edge("memory", "plan")

    graph.add_edge("plan", "critic")

    graph.add_conditional_edges(
        "critic",
        route_plan_step,
        {
            "execute": "execute",
            "reason": "reason",
            "llm": "llm",
            "plan": "plan",
            "replanning": "replanning"
        }
    )


    graph.add_edge("reason", "agent")
    graph.add_edge("llm", "update_progress")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "memory_store": "memory_store"
        }
    )

    graph.add_conditional_edges(
        "execute",
        should_replan,
        {
            "replanning": "replanning",
            "reason": "update_progress"
        }
    )

    graph.add_conditional_edges(
        "tools",
        should_replan,
        {
            "replanning": "replanning",
            "reason": "update_progress"
        }
    )

    graph.add_conditional_edges(
        "update_progress",
        route_plan_step,
        {
            "execute": "execute",
            "reason": "reason",
            "llm": "llm",
            "memory_store": "memory_store"
        }
    )

    graph.add_conditional_edges(
        "replanning",
        route_after_replanning,
        {
            "critic": "critic",
            "execute": "execute",
            "reason": "reason",
            "llm": "llm",
            "memory_store": "memory_store"
        }
    )

    graph.add_edge("memory_store", END)


    agent = graph.compile()

    return agent