from AgentApp.graph.state import AgentState
from memory.LongTerm import LongTermMemory

def memory_node(state: AgentState):

    query = state["user_goal"]

    memory = LongTermMemory()

    experiences = memory.retrieve(query, memoryKey="experience", retrievalTopK=5, finalTopK=3)
    profile = memory.retrieve(query, memoryKey="user_profile", retrievalTopK=5, finalTopK=2)

    return {
        "retrieved_memory": experiences + profile
    }