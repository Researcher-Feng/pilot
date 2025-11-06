import os
from v1.utils.api_config import key_config, key_url, ds_key_url, smith_key
selected_model = 'deepseek-official'
os.environ["OPENAI_API_KEY"] = key_config[selected_model]
os.environ["OPENAI_BASE_URL"] = key_url
os.environ["DEEPSEEK_API_KEY"] = key_config[selected_model]
os.environ["DEEPSEEK_BASE_URL"] = ds_key_url
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_API_KEY"] = smith_key

from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, wrap_tool_call, dynamic_prompt, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware

import sys
sys.path.append(r'/v1/prompt')
sys.path.append(r'/v1/function')
from v1.prompt.system import *
from v1.function.contex_fun import *
from v1.function.format_fun import *
from v1.function.memory_fun import *


class SimpleAgent(object):
    """Agent class."""
    def __init__(self):
        """Initialize agent."""
        self.prompt_sys = None
        self.tools = []
        self.context_schema = None
        self.response_format = None
        self.checkpointer = None
        self.middleware_list = []

        self.agent_config = None
        self.response = None
        self.context = None

        self.middleware_er = None
        self.custom_middleware_er = CustomMiddleware
        self.basic_model = None
        self.advanced_model = None
        self.agent = None

    def model_init(self, model_name, model_type='basic', **kwargs):
        if model_type == 'basic':
            self.basic_model = init_chat_model(
                model_name,
                **kwargs  # temperature=0.5, timeout=10, max_tokens=1000
            )
        elif model_type == 'advanced':
            self.advanced_model = init_chat_model(
                model_name,
                **kwargs  # temperature=0.5, timeout=300, max_tokens=4096
            )
        else:
            raise NotImplementedError
        self.middleware_er = MiddlewareFunc(self.basic_model, self.advanced_model)

    def agent_init(self, prompt_sys_name=None, tools_list=None, context_schema=Context,
                  response_format=ResponseFormat, checkpointer=simple_checkpointer, middleware=None):
        if prompt_sys_name:
            self.prompt_sys = prompt_sys_name
        if tools_list:
            self.tools = tools_list
        self.context_schema = context_schema
        self.response_format = response_format
        self.checkpointer = checkpointer
        if middleware:
            for m in middleware:
                if m == 'dynamic':
                    self.middleware_list.append(self.middleware_er.middleware_dynamic_model_selection())
                elif m == 'handle_tool_errors':
                    self.middleware_list.append(self.middleware_er.middleware_handle_tool_errors())
                elif m == 'user_role_prompt':
                    self.middleware_list.append(self.middleware_er.middleware_user_role_prompt())
                elif m == 'CustomMiddleware':
                    self.middleware_list.append(self.custom_middleware_er())
        self.agent = create_agent(
            model=self.basic_model,
            system_prompt=self.prompt_sys,
            tools=self.tools,
            context_schema=self.context_schema,
            response_format=self.response_format,
            checkpointer=self.checkpointer,
            middleware=self.middleware_list,
        )

    def config_create(self, key_i, value_i):
        self.agent_config = {"configurable": {key_i: value_i}}

    def context_set(self, **kwargs):
        self.context = self.context_schema(**kwargs)

    def chat_once(self, user_input, response_type='invoke', silence=False, **kwargs):
        if response_type == 'invoke':
            self.agent_response_invoke(user_input, **kwargs)
            if not silence:
                self.agent_output()
        else:
            self.agent_response_stream(user_input, **kwargs)

    def agent_response_invoke(self, user_input, **kwargs):
        self.response = self.agent.invoke({
            "messages": [
                {"role": "user",
                 "content": user_input}
                ]
            },
            config=self.agent_config,
            context=self.context,
            **kwargs
        )

    def agent_response_stream(self, user_input, **kwargs):
        for chunk in self.agent.stream({
                "messages": [
                    {"role": "user",
                     "content": user_input}
                    ]
                },
                config=self.agent_config,
                context=self.context,
                stream_mode="values",
                **kwargs
        ):
            # Each chunk contains the full state at that point
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                print(f"Agent: {latest_message.content}")
            elif latest_message.tool_calls:
                print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")

    def agent_output(self):
        for i in self.response['messages']:
            print(f"{type(i)}: {i.content}")
        # print(self.response[output_field_name])
        # print(self.response['structured_response'].main_response)


class CustomState(AgentState):
    user_preferences: dict


class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState # 这个赋值是给框架使用的

    def before_model(self, state: CustomState, runtime):
        # 基于记忆调整系统提示
        if state.user_preferences.get("style") == "technical":
            return {"system_prompt": "Provide detailed technical explanations with specifications."}
        else:
            return {"system_prompt": "Explain concepts in simple, easy-to-understand terms."}
    def before_tool_call(self, state: CustomState, runtime, tool_name):
        # 基于用户偏好选择不同的搜索工具
        if state.user_preferences.get("verbosity") == "detailed":
            return "technical_search"  # 使用提供详细结果的技术搜索
        else:
            return "simple_search"     # 使用简化的搜索


class MiddlewareFunc(object):
    def __init__(self, basic_model, advanced_model):
        self.basic_model = basic_model
        self.advanced_model = advanced_model

    def middleware_handle_tool_errors(self) -> Callable:
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
        return handle_tool_errors

    def middleware_dynamic_model_selection(self) -> Callable:
        """返回一个配置好的中间件函数"""

        @wrap_model_call
        def dynamic_model_middleware(request: ModelRequest, handler) -> ModelResponse:
            """Choose model based on conversation complexity."""
            messages = request.state.get("messages", [])
            message_count = len(messages)

            # 调试信息 - 查看当前消息数量
            print(f"Debug: Current message count = {message_count}")
            # print(f"Debug: Messages: {[msg.get('content', '')[:50] for msg in messages]}")  # 打印前50个字符

            # 条件1：基于消息数量（降低阈值）
            if message_count >= 2:  # 从第2条消息开始使用高级模型
                # 条件2：基于消息内容复杂度
                user_content = ""
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        user_content = msg.content
                        break

                # 检测复杂问题关键词
                complex_keywords = ['solution', 'problem', 'explain', 'analyze', 'how to', 'why', 'compare']
                is_complex = any(keyword in user_content.lower() for keyword in complex_keywords)

                if is_complex and self.advanced_model:
                    model = self.advanced_model
                    model_name = self.advanced_model.model_name
                    print(f'🔀 Selected ADVANCED model for complex query: {model_name}')
                else:
                    model = self.basic_model
                    model_name = self.basic_model.model_name
                    print(f'🔀 Selected BASIC model: {model_name}')
            else:
                model = self.basic_model
                model_name = self.basic_model.model_name
                print(f'🔀 Selected BASIC model (first message): {model_name}')

            request.model = model
            return handler(request)

        return dynamic_model_middleware

    def middleware_user_role_prompt(self) -> Callable:
        @dynamic_prompt
        def user_role_prompt(request: ModelRequest) -> str:
            """Generate system prompt based on user role."""
            user_role = request.runtime.context.user_role

            if user_role == "expert":
                return EXPECT_PROMPT
            elif user_role == "beginner":
                return BEGINNER_PROMPT

            return BASE_PROMPT
        return user_role_prompt






