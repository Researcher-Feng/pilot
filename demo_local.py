from langchain.agents import create_agent
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:0.5b",
    temperature=0,
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

