from AgentApp.graph.state import AgentState
from AgentApp.utils.json_parsing import safe_json_load
from memory.LongTerm import LongTermMemory
from AgentApp.initialize_llm import get_llm

def memory_store_node(state:AgentState):

    llm = get_llm()
    prompt = f"""
        You are a memory extraction system.

        Your job is to extract information that is useful for future interactions.

        You may only create memories in two categories:

        user_profile:
        Who is the user and what stable information about them
        could help future interactions?
        This information must be about user information, his preference, his skills, his background and anything to the user specifically.

        experience:
        A reusable lesson about agent behavior, planning,
        replanning, tool usage, dependency management,
        memory retrieval, or execution strategy.

        The lesson must describe something the agent learned
        about HOW TO SOLVE TASKS better.

        It must NOT describe:

        - facts about the user
        - facts about a project
        - facts about a repository
        - facts discovered from tools
        - domain knowledge

        IMPORTANT:
        Do not summarize the task.
        Do not store project facts merely because they are interesting.
        Do not store tool outputs.
        Do not store temporary execution details.
        For experience memories, generalize concrete events into reusable lessons.
        Each memory must only contain an atomic info not a mix of knowledge.

        The importance score should be calculated using the following parameters:
        - Does it make the agent know the user better?
        - Does it make planning, replanning, or tool selection better by reducing round-around trips? (Are there any avoidable issues in this run based on tool results and planning?)
        - Does it contain any info about tool outputs? (negative score)
        - Is this memory a generalization that is reusable (positive score) or specific and overly detailed? (negative score) 

        User goal:
        {state["user_goal"]}

        Planning history:
        {state["planning_history"]}

        Replanning history (if occurred):
        {state["replanning_history"]}

        Conversation:
        {state["messages"]}

        Final result:
        {state['variables']['final_outcome']}

        Return JSON only:

        {{
            "memories": [
                {{
                    "memory_type": "user_profile",
                    "content": "The user prefers Tensorflow over Pytorch",
                    "importance": 1-10
                }},

                {{
                    "memory_type": "user_profile",
                    "content": "The user prefers building ML projects with practical retrieval-based architectures.",
                    "importance": 7 (Good to know user's projects, but it's not deterministic through only one run )
                }},

                {{
                    "memory_type": "experience",
                    "content": "Always use tool X before tool Y in order to avoid critic planning issue or replanning",
                    "importance": 10
                }}
            ]
        }}
        """
    
    result = llm.invoke(prompt)

    print("\n=== MEMORY EXTRACTION ===")
    print(result.content)

    memories = safe_json_load(result.content)["memories"]

    memory_engine = LongTermMemory()

    for memory in memories:
        memory_item = {
            "content": memory["content"],
            "memory_type": memory["memory_type"],
            "source": "memory_extraction",
            "metadata": {
                "importance": memory["importance"],
                "goal": state["user_goal"]
            }
        }

        memory_engine.store(memory_item)

    print("Successfully stored")
