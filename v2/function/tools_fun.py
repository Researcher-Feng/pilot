from langchain.tools import tool, ToolRuntime
from v2.function.contex_fun import Context


@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve user information based on user ID."""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def parallel_thinking(query: str) -> str:
    """Conduct parallel thinking for the problem."""
    pass
