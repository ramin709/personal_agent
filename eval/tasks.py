TASKS = [

    {
        "id": "weather_01",
        "input": "What is the weather in Berlin?",
        "expected_tools": [
            {"name": "get_weather", "args": {"city": "Berlin"}}
        ],
        "expected_answer": "18°C"
    },

    {
        "id": "math_01",
        "input": "What is 15% of 200?",
        "expected_tools": [
            {
                "name": "calculator",
                "args": {"expression": "15% * 200"}
            }
        ],
        "expected_answer": "30"
    },

    {
        "id": "weather_03",
        "input": "Which is warmer, Berlin or London?",
        "expected_tools": [
            {"name": "get_weather", "args": {"city": "Berlin"}},
            {"name": "get_weather", "args": {"city": "London"}}
        ],
        "expected_answer": "Berlin"
    },

]