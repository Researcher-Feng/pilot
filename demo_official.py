from langchain_deepseek import ChatDeepSeek
from v1.function.contex_fun import Context
import os
from v1.utils.api_config import key_config, key_url, ds_key_url, smith_key
selected_model = 'deepseek-official'
os.environ["OPENAI_API_KEY"] = key_config[selected_model]
os.environ["OPENAI_BASE_URL"] = key_url
os.environ["DEEPSEEK_API_KEY"] = key_config[selected_model]
os.environ["DEEPSEEK_BASE_URL"] = ds_key_url
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_API_KEY"] = smith_key

basic_model = ChatDeepSeek(model="deepseek-chat")
advanced_model = ChatDeepSeek(model="deepseek-reasoner")

from langchain.tools import tool


@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Results for: {query}"

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"Weather in {location}: Sunny, 72°F"

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call, dynamic_prompt, ModelRequest
from langchain_core.messages import ToolMessage


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

@dynamic_prompt
def user_role_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    user_role = request.runtime.context.user_role
    base_prompt = "You are a helpful assistant."

    if user_role == "expert":
        return f"{base_prompt} Provide detailed technical responses."
    elif user_role == "beginner":
        return f"{base_prompt} Explain concepts simply and avoid jargon."

    return base_prompt


agent = create_agent(
    basic_model,
    tools=[search, get_weather],
    middleware=[handle_tool_errors, user_role_prompt],
    context_schema=Context
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Explain machine learning"}]},
    config={"configurable": {"thread_id": "1"}},
    context=Context(user_id="1", user_role="expert")
)
for i in result['messages']:
    print(f"{type(i)}: {i.content}")



