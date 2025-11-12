# agent_IO.py
import os
import sys
sys.path.append(r'/v2/prompt')
sys.path.append(r'/v2/function')
from v2.function import *
from v2.prompt import *
from v2.utils import *
selected_model = 'deepseek-official'
os.environ["OPENAI_API_KEY"] = key_config.get(selected_model, "")
os.environ["OPENAI_BASE_URL"] = key_url
os.environ["DEEPSEEK_API_KEY"] = key_config.get(selected_model, "")
os.environ["DEEPSEEK_BASE_URL"] = ds_key_url
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_API_KEY"] = smith_key

from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, wrap_tool_call, dynamic_prompt, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Callable


class ModelConfig:
    """统一的模型配置类"""

    def __init__(self, model_type: str = "api", **kwargs):
        self.model_type = model_type  # "api" 或 "local"
        self.model_name = kwargs.get("model_name", "")
        self.base_url = kwargs.get("base_url", "")
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 2000)
        self.timeout = kwargs.get("timeout", 30)

        # 本地模型特定配置
        if model_type == "local":
            self.base_url = kwargs.get("base_url", "http://localhost:11434")
            self.extra_params = kwargs.get("extra_params", {})


class SimpleAgent(object):
    """增强的Agent类，支持多模式和本地调用"""

    def __init__(self, agent_type: str = "student"):
        """初始化agent

        Args:
            agent_type: "student" 或 "teacher"
        """
        self.agent_type = agent_type
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
        self.model = None
        self.agent = None

        # 多智能体系统配置
        self.dialogue_history = []
        self.max_turns = 5
        self.current_turn = 0

    def model_init(self, model_config: ModelConfig):
        """初始化模型，支持API和本地调用"""
        common_kwargs = {
            "temperature": model_config.temperature,
            "timeout": model_config.timeout,
            "max_tokens": model_config.max_tokens,
        }

        if model_config.model_type == "local":
            # 本地模型配置（如Ollama）
            local_kwargs = {
                "base_url": model_config.base_url,
                **model_config.extra_params,
                **common_kwargs
            }
            self.model = init_chat_model(model_config.model_name, **local_kwargs)
            print(f"✅ 初始化本地模型: {model_config.model_name}")
        else:
            # API模型配置
            api_kwargs = common_kwargs
            self.model = init_chat_model(model_config.model_name, **api_kwargs)
            print(f"✅ 初始化API模型: {model_config.model_name}")

        self.middleware_er = MiddlewareFunc(self.model, self.model)  # 暂时使用相同模型

    def agent_init(self, prompt_sys_name=None, tools_list=None, context_schema=Context,
                   response_format=ResponseFormat, checkpointer=simple_checkpointer,
                   middleware=None, **kwargs):
        """初始化agent配置"""
        if prompt_sys_name:
            self.prompt_sys = prompt_sys_name
        if tools_list:
            self.tools = tools_list
        self.context_schema = context_schema
        self.response_format = response_format
        self.checkpointer = checkpointer

        # 多智能体系统参数
        self.max_turns = kwargs.get("max_turns", 5)
        self.parallel_thinking = kwargs.get("parallel_thinking", False)
        self.socratic_teaching = kwargs.get("socratic_teaching", False)
        self.math_background = kwargs.get("math_background", "intermediate")

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
            model=self.model,
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

    def multi_agent_chat(self, teacher_agent, problem, **kwargs):
        """多智能体对话 - 显式交互模式"""
        print(f"\n🎯 开始解题: {problem}")
        print("=" * 50)

        student_response = self._invoke_agent(problem)
        self.dialogue_history.append(("student", student_response))

        for turn in range(self.max_turns):
            print(f"\n🔄 第 {turn + 1} 轮对话:")
            print("-" * 30)

            # 教师回应
            teacher_input = self._format_teacher_input(problem, self.dialogue_history)
            teacher_response = teacher_agent._invoke_agent(teacher_input)
            self.dialogue_history.append(("teacher", teacher_response))

            print(f"👨‍🏫 教师: {teacher_response}")

            # 检查是否应该结束对话
            if self._teacher_end_dialogue(teacher_response):
                print("✅ 教师认为解题完成")
                break

            # 学生回应
            student_input = self._format_student_input(problem, self.dialogue_history)
            student_response = self._invoke_agent(student_input)
            self.dialogue_history.append(("student", student_response))

            print(f"👨‍🎓 学生: {student_response}")

            # 检查学生是否得出正确答案
            if self._student_end_dialogue(student_response, problem):
                print("🎉 学生得出正确答案!")
                break

            self.current_turn = turn + 1

        return self._get_final_answer()

    def _invoke_agent(self, input_text):
        """调用agent并返回响应"""
        response = self.agent.invoke({
            "messages": [{"role": "user", "content": input_text}]
        }, config=self.agent_config, context=self.context)

        if response and 'messages' in response:
            return response['structured_response'].main_response
        return ""

    def _format_teacher_input(self, problem, history):
        """格式化教师输入"""
        context = f"Question: {problem}\n\n"
        for role, content in history:
            context += f"{role}: {content}\n\n" # 下面的英文：请根据以上对话继续引导学生解题:
        context += "Please continue guiding students through the problem-solving process based on the above dialogue:"
        return context

    def _format_student_input(self, problem, history):
        """格式化学生输入"""
        context = f"Raw question: {problem}\n\nChat history:\n"
        for role, content in history:
            context += f"{role}: {content}\n\n"
        context += "Please continue solving the problem or respond to the teacher's instructions:"
        return context

    def _teacher_end_dialogue(self, teacher_response):
        """判断是否应该结束对话"""
        end_phrases = ["完成", "finish"]
        return any(phrase in teacher_response.lower() for phrase in end_phrases)

    def _student_end_dialogue(self, student_response, problem):
        """简单判断学生是否得出正确答案"""
        # 这里可以集成更复杂的答案验证逻辑
        math_indicators = ["答案是", "结果为", "等于", "=", "final answer is"]
        return any(indicator in student_response for indicator in math_indicators)

    def _get_final_answer(self):
        """获取最终答案"""
        if self.dialogue_history:
            return self.dialogue_history[-1][1]
        return ""

    def agent_response_invoke(self, user_input, **kwargs):
        self.response = self.agent.invoke({
            "messages": [{"role": "user", "content": user_input}]
        }, config=self.agent_config, context=self.context, **kwargs)

    def agent_response_stream(self, user_input, **kwargs):
        for chunk in self.agent.stream({
            "messages": [{"role": "user", "content": user_input}]
        }, config=self.agent_config, context=self.context, stream_mode="values", **kwargs):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                print(f"Agent: {latest_message.content}")
            elif latest_message.tool_calls:
                print(f"Calling tools: {[tc['name'] for tc in latest_message.tool_calls]}")

    def agent_output(self):
        for i in self.response['messages']:
            print(f"{type(i)}: {i.content}")


# 其他类保持不变...
class CustomState(AgentState):
    user_preferences: dict


class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState

    def before_model(self, state: CustomState, runtime):
        if state.user_preferences.get("style") == "technical":
            return {"system_prompt": "Provide detailed technical explanations with specifications."}
        else:
            return {"system_prompt": "Explain concepts in simple, easy-to-understand terms."}

    def before_tool_call(self, state: CustomState, runtime, tool_name):
        if state.user_preferences.get("verbosity") == "detailed":
            return "technical_search"
        else:
            return "simple_search"


class MiddlewareFunc(object):
    def __init__(self, basic_model, advanced_model):
        self.basic_model = basic_model
        self.advanced_model = advanced_model

    def middleware_handle_tool_errors(self) -> Callable:
        @wrap_tool_call
        def handle_tool_errors(request, handler):
            try:
                return handler(request)
            except Exception as e:
                return ToolMessage(
                    content=f"Tool error: Please check your input and try again. ({str(e)})",
                    tool_call_id=request.tool_call["id"]
                )

        return handle_tool_errors

    def middleware_dynamic_model_selection(self) -> Callable:
        @wrap_model_call
        def dynamic_model_middleware(request: ModelRequest, handler) -> ModelResponse:
            messages = request.state.get("messages", [])
            message_count = len(messages)
            print(f"Debug: Current message count = {message_count}")

            if message_count >= 2:
                user_content = ""
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        user_content = msg.content
                        break
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
            user_role = request.runtime.context.user_role
            if user_role == "expert":
                return EXPECT_PROMPT
            elif user_role == "beginner":
                return BEGINNER_PROMPT
            return BASE_PROMPT

        return user_role_prompt