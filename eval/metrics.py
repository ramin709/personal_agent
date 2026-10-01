def extract_tool_calls(result):
    calls = []

    for message in result["messages"]:
        if hasattr(message, "tool_calls"):
            for call in message.tool_calls:
                calls.append({
                    "name": call["name"],
                    "args": call["args"]
                })

    return calls

def tool_selection_accuracy(actual, expected):

    expected_names = [x["name"] for x in expected]
    actual_names = [x["name"] for x in actual]

    correct = sum(
        name in actual_names
        for name in expected_names
    )

    if not expected_names:
        return 1.0

    return correct / len(expected_names)

def tool_argument_accuracy(actual, expected):

    correct = 0

    for exp in expected:

        for act in actual:

            if (
                exp["name"] == act["name"]
                and exp["args"] == act["args"]
            ):
                correct += 1
                break

    if not expected:
        return 1.0

    return correct / len(expected)

def get_steps(result):
    return result["current_step"]

def get_tool_call_count(result):

    count = 0

    for message in result["messages"]:

        if hasattr(message, "tool_calls"):
            count += len(message.tool_calls)

    return count

def get_final_answer(result):

    for message in reversed(result["messages"]):

        if message.__class__.__name__ == "AIMessage":
            if not getattr(message, "tool_calls", None):
                return message.content

    return ""

