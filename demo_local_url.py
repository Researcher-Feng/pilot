from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

# ERROR
# model = init_chat_model('qwen2.5:0.5b', base_url='http://localhost:11434')
model = ChatOllama(
    model='qwen2.5:1.5b-Mine',
    base_url=r'http://localhost:11434',
    temperature=0.7,
    num_predict=2000,
)
print('Success model')
agent = create_agent(model=model)

messages = {
            "messages": [{"role": "user", "content": "You are a helpful assistant. Please introduce deep learning."}]
        }
ai_msg = agent.invoke(messages)
# print(ai_msg)
print(ai_msg['messages'][-1].content)

