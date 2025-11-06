from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 加载.env文件的环境变量
load_dotenv()


# 创建一个大语言模型，model指定了大语言模型的种类
model = ChatOpenAI(model="deepseek-chat")
# 定义传递给模型的消息队列
# SystemMessage的content指定了大语言模型的身份，即他应该做什么，对他进行设定
# HumanMessage的content是我们要对大语言模型说的话，即用户的输入
messages = [
    SystemMessage(content="把下面的语句翻译为英文。"),
    HumanMessage(content="今天天气怎么样？"),
]
# 打印模型的输出结果
print(model.invoke(messages).content)


# 使用result接收模型的输出，result就是一个AIMessage对象
result = model.invoke(messages)
# 定义一个解析器对象
parser = StrOutputParser()
# 使用解析器对result进行解析
parser.invoke(result)



prompt_template = ChatPromptTemplate.from_messages([
    ("system", "把下面的语句翻译为{language}。"),
    ("user", "{text}")]
)
prompt = prompt_template.invoke({"language": "英文", "text": "今天天气怎么样？"})
print(prompt)


# 安装并引入需要的包
from fastapi import FastAPI
from langserve import add_routes
import uvicorn


chain = prompt_template | model | parser
print(chain.invoke({"language": "英文", "text": "今天天气怎么样？"}))
# 使用FastAPI创建一个可访问的应用
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)
# 将该应用包装为一个服务
add_routes(
    app,
    chain,
    path="/chain",
)
# 通过unicorn启动服务
uvicorn.run(app, host="localhost", port=8000)




