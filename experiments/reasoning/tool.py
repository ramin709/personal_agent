from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """ Evaluate a mathematical expression """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error {e}"
    
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""

    weather_data = {
        "Berlin": "18°C, cloudy",
        "Baku": "27°C, sunny",
        "London": "15°C, rainy",
    }

    return weather_data.get(
        city,
        f"No weather data available for {city}"
    )