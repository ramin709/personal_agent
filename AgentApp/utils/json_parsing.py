import json
import re

def safe_json_load(text: str):
    text = text.strip()

    # Remove markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    text = text.strip()

    # Try parsing the whole response first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # If the model added prose, extract the JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError("No valid JSON object found in LLM response.")

    json_text = match.group(0)

    json_text = json_text.replace(": True", ": true")
    json_text = json_text.replace(": False", ": false")
    json_text = json_text.replace(": None", ": null")

    return json.loads(json_text)
