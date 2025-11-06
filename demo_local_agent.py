from langchain.agents import create_agent
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:0.5b",
    temperature=0,
    # other params...
)

agent = create_agent(llm)
response = agent.invoke({
            "messages": [{"role": "user", "content": "You are a helpful assistant that translates English to French. Translate the user sentence."}]
        })

print(response)

