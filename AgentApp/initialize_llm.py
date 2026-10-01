from pathlib import Path
from langchain_openai import ChatOpenAI
import os
from functools import lru_cache
from dotenv import load_dotenv

@lru_cache
def get_llm():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    llm = ChatOpenAI(
        model="inclusionai/ling-3.0-flash-sante:free",
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_KEY")
    )

    return llm