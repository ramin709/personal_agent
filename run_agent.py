from AgentApp.graph.agent_graph import build_graph

def main():
    agent = build_graph()

    user_question = (
        "Examine my NLP ReceipeQA project on my github account and explain its architecture to me"
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

        "plan": [],

        "critic_comment": {},

        "current_plan_step": 0,

        "current_step": 0,

        "completed_steps": [],

        "variables": {},

        "plan_origin": "",

        "plan_modified": False,

        "retrieved_memory": [],

        "planning_history": [],
        "replanning_history": []

    })


    print("\n=== FINAL ANSWER ===")
    print(result["variables"]["final_outcome"])

if __name__ == "__main__":
    main()