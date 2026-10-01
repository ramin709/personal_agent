# eval/runner.py

import time
from eval.tasks import TASKS
from eval.metrics import (
    extract_tool_calls,
    tool_selection_accuracy,
    tool_argument_accuracy,
    get_steps,
    get_tool_call_count,
    get_final_answer,
)

def answer_contains_expected(answer, expected):
    return str(expected).lower() in answer.lower()

def run_task(task, agent):

    start = time.perf_counter()

    result = agent.invoke({
        "messages": [
            ("user", task["input"])
        ],

        "user_goal": task["input"],

        "reasoning": "",

        "current_step": 0,

        "completed_steps": []
    })

    latency = time.perf_counter() - start

    actual_tools = extract_tool_calls(result)

    answer = get_final_answer(result)

    return {
        "task_id": task["id"],
        "answer": answer,

        "success": answer_contains_expected(
            answer,
            task["expected_answer"]
        ),

        "tool_selection_accuracy":
            tool_selection_accuracy(
                actual_tools,
                task["expected_tools"]
            ),

        "tool_argument_accuracy":
            tool_argument_accuracy(
                actual_tools,
                task["expected_tools"]
            ),

        "steps": get_steps(result),
        "tool_calls": get_tool_call_count(result),
        "latency": latency,

        "tokens": sum(
            m.usage_metadata.get("total_tokens", 0)
            for m in result["messages"]
            if hasattr(m, "usage_metadata")
            and m.usage_metadata
        ),

        "raw_result": result
    }


def run_benchmark(agent):

    results = []

    for task in TASKS:

        print(f"\nRunning {task['id']}...")

        result = run_task(task , agent)

        results.append(result)

        print(result)

    return results

def summarize(results):

    n = len(results)

    return {
        "success_rate":
            sum(r["success"] for r in results) / n,

        "tool_selection_accuracy":
            sum(
                r["tool_selection_accuracy"]
                for r in results
            ) / n,

        "tool_argument_accuracy":
            sum(
                r["tool_argument_accuracy"]
                for r in results
            ) / n,

        "avg_steps":
            sum(r["steps"] for r in results) / n,

        "avg_tool_calls":
            sum(r["tool_calls"] for r in results) / n,

        "avg_latency":
            sum(r["latency"] for r in results) / n,

        "avg_tokens":
            sum(r["tokens"] for r in results) / n,
    }