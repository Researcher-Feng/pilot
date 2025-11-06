import os
from v1.utils.api_config import key_config, key_url, ds_key_url, smith_key
selected_model = 'deepseek-official'
os.environ["OPENAI_API_KEY"] = key_config[selected_model]
os.environ["OPENAI_BASE_URL"] = key_url
os.environ["DEEPSEEK_API_KEY"] = key_config[selected_model]
os.environ["DEEPSEEK_BASE_URL"] = ds_key_url
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_API_KEY"] = smith_key

from langchain_deepseek import ChatDeepSeek


llm = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)



