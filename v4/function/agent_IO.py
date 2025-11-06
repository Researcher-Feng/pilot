# agent_IO.py
import logging
import os
import sys
sys.path.append(r'../..')
from typing import Callable, List, Optional, Dict, Any
from utils.api_config import *
from utils.evaluator import extract_answer, logger
from utils.MARIO_EVAL.demo import is_equiv_MATH
from prompt.system import *
from function.contex_fun import *
from function.format_fun import *
from function.memory_fun import *
selected_model = 'deepseek-official'
os.environ["OPENAI_API_KEY"] = key_config.get(selected_model, "")
os.environ["OPENAI_BASE_URL"] = key_url
os.environ["DEEPSEEK_API_KEY"] = key_config.get(selected_model, "")
os.environ["DEEPSEEK_BASE_URL"] = ds_key_url
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_API_KEY"] = smith_key
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('ALL_PROXY', None)
os.environ.pop('all_proxy', None)

from langchain_core.messages import HumanMessage, ToolMessage
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, wrap_tool_call, dynamic_prompt, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
import datetime
import sys

sys.path.append(r'D:\DeepLearning\Code\LangChain\prompt')
sys.path.append(r'D:\DeepLearning\Code\LangChain\function')
from prompt.system import *
from function.tools_fun import *
from function.contex_fun import *
from function.format_fun import *
from function.memory_fun import *


class DialogueRecord:
    """对话记录类"""

    def __init__(self, problem: str, correct_answer: str, debug_mode: bool = False):
        self.problem = problem
        self.correct_answer = correct_answer
        self.debug_mode = debug_mode
        self.turns = []
        self.final_student_answer = ""
        self.first_correct = False
        self.correct = False
        self.leaked_answer = False
        self.parallel_thinking_count = 0
        self.thinking_paths_count = 0
        self.total_turns = 0

    def add_turn(self, turn_data: Dict[str, Any]):
        """添加一轮对话记录"""
        self.turns.append(turn_data)
        self.total_turns = len(self.turns)

    def analyze_student_response(self, response: str):
        """分析学生回复"""
        # 统计并行思考标签
        parallel_count = response.count('<Parallel>')
        self.parallel_thinking_count += parallel_count

        # 统计思考路径标签
        path_count = response.count('<Path>')
        self.thinking_paths_count += path_count

        return parallel_count, path_count

    def check_answer_leakage(self, teacher_response: str):
        """检查教师是否泄露答案"""
        # 简单的答案泄露检测逻辑
        leakage_indicators = [
            "the answer is " + self.correct_answer,
            "the result is " + self.correct_answer,
            "equals to " + self.correct_answer,
            "= " + self.correct_answer
        ]

        leakage_detected = any(
            indicator.lower() in teacher_response.lower()
            for indicator in leakage_indicators
            if indicator.strip()
        )

        if leakage_detected:
            self.leaked_answer = True

        return leakage_detected

    def to_dict(self):
        """转换为字典格式"""
        return {
            "problem": self.problem,
            "correct_answer": self.correct_answer,
            "final_student_answer": self.final_student_answer,
            "correct": self.correct,
            "leaked_answer": self.leaked_answer,
            "parallel_thinking_count": self.parallel_thinking_count,
            "thinking_paths_count": self.thinking_paths_count,
            "total_turns": self.total_turns,
            "turns": self.turns
        }


class ExperimentRecorder:
    """实验记录器"""

    def __init__(self, experiment_name: str = "multi_agent_math"):
        self.experiment_name = experiment_name
        self.records = []
        self.summary_stats = {}
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_record(self, record: DialogueRecord):
        """添加对话记录"""
        self.records.append(record)

    def calculate_statistics(self):
        """计算统计信息"""
        if not self.records:
            return {}

        total_problems = len(self.records)
        correct_answers = sum(1 for r in self.records if r.correct)
        leaked_answers = sum(1 for r in self.records if r.leaked_answer)
        total_turns = sum(r.total_turns for r in self.records)
        total_parallel_thinking = sum(r.parallel_thinking_count for r in self.records)
        total_thinking_paths = sum(r.thinking_paths_count for r in self.records)

        self.summary_stats = {
            "total_problems": total_problems,
            "accuracy": correct_answers / total_problems if total_problems > 0 else 0,
            "answer_leakage_rate": leaked_answers / total_problems if total_problems > 0 else 0,
            "avg_turns_per_problem": total_turns / total_problems if total_problems > 0 else 0,
            "avg_parallel_thinking": total_parallel_thinking / total_problems if total_problems > 0 else 0,
            "avg_thinking_paths": total_thinking_paths / total_problems if total_problems > 0 else 0,
            "correct_answers": correct_answers,
            "leaked_answers": leaked_answers
        }

        return self.summary_stats

    def save_results(self, output_dir: str = "results"):
        """保存结果到文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 保存详细记录
        detailed_data = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "records": [record.to_dict() for record in self.records],
            "summary": self.summary_stats
        }

        return detailed_data, self.summary_stats

    def print_summary(self):
        """打印实验摘要"""
        if not self.summary_stats:
            self.calculate_statistics()

        print("\n" + "=" * 60)
        print("🎯 实验摘要统计")
        print("=" * 60)
        print(f"总问题数: {self.summary_stats['total_problems']}")
        print(f"准确率: {self.summary_stats['accuracy']:.4f}")
        print(f"答案泄露率: {self.summary_stats['answer_leakage_rate']:.4f}")
        print(f"平均对话轮数: {self.summary_stats['avg_turns_per_problem']:.2f}")
        print(f"平均并行思考次数: {self.summary_stats['avg_parallel_thinking']:.2f}")
        print(f"平均思考路径数: {self.summary_stats['avg_thinking_paths']:.2f}")
        print(f"正确答案数: {self.summary_stats['correct_answers']}")
        print(f"泄露答案数: {self.summary_stats['leaked_answers']}")
        print("=" * 60)


class ModelConfig:
    """统一的模型配置类"""

    def __init__(self, model_type: str = "api", **kwargs):
        self.model_type = model_type  # "api" 或 "local"
        self.model_name = kwargs.get("model_name", "")
        self.base_url = kwargs.get("base_url", "http://localhost:11434")
        self.temperature = kwargs.get("temperature", 0.7)
        self.max_tokens = kwargs.get("max_tokens", 2000)
        self.timeout = kwargs.get("timeout", 30)
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
        self.correct_answer = None
        self.student_answer = None

        # 功能开关
        self.parallel_thinking_enabled = False
        self.socratic_teaching_enabled = False
        self.math_background_level = "intermediate"

    def model_init(self, model_config: ModelConfig):
        """初始化模型，支持API和本地调用"""
        if model_config.model_type == "local":
            # 使用 Ollama 本地模型
            self.model = ChatOllama(
                model=model_config.model_name,
                base_url=model_config.base_url,
                temperature=model_config.temperature,
                num_predict=model_config.max_tokens,
                **model_config.extra_params
            )
            print(f"✅ 初始化本地模型: {model_config.model_name}")
        else:
            # API模型配置
            api_kwargs = {
                "temperature": model_config.temperature,
                "timeout": model_config.timeout,
                "max_tokens": model_config.max_tokens,
            }
            self.model = init_chat_model(model_config.model_name, **api_kwargs)
            print(f"✅ 初始化API模型: {model_config.model_name}")

        self.middleware_er = MiddlewareFunc(self.model, self.model)

    def set_correct_answer(self, correct_answer: str):
        """设置正确答案（教师agent使用）"""
        self.correct_answer = correct_answer

    def agent_init(self, model_config, prompt_sys_name=None, tools_list=None, context_schema=Context,
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

        # 设置功能开关
        self.max_turns = kwargs.get("max_turns", 5)
        self.parallel_thinking_enabled = kwargs.get("parallel_thinking", False)
        self.socratic_teaching_enabled = kwargs.get("socratic_teaching", False)
        self.math_background_level = kwargs.get("math_background", "intermediate")

        # 根据功能开关调整系统提示词
        self._adjust_prompt_based_on_settings()

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

        if model_config.model_type == "local":
            self.agent = self._create_simple_agent()
        else:
            self.agent = create_agent(
                model=self.model,
                system_prompt=self.prompt_sys,
                tools=self.tools,
                context_schema=self.context_schema,
                response_format=self.response_format,
                checkpointer=self.checkpointer,
                middleware=self.middleware_list,
            )

    def _create_simple_agent(self):
        """创建不包含工具的简单对话agent"""
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.messages import AIMessage, HumanMessage

        # 创建简单的对话链
        prompt = ChatPromptTemplate.from_messages([
            ("system", self.prompt_sys),
            ("user", "{input}")
        ])

        # 简单的对话链，不包含任何工具
        chain = prompt | self.model

        # 包装成类似agent的接口
        class SimpleAgentWrapper:
            def __init__(self, chain):
                self.chain = chain

            def invoke(self, input_dict, config=None, context=None):
                user_input = input_dict["messages"][-1]['content']
                # 调用模型
                response_content = self.chain.invoke({"input": user_input})

                # 确保response_content是字符串
                if hasattr(response_content, 'content'):
                    response_text = response_content.content
                else:
                    response_text = str(response_content)

                # 返回与LangGraph兼容的格式
                return response_text

            def stream(self, input_dict, config=None, context=None, stream_mode="values"):
                # 简单的流式响应实现
                result = self.invoke(input_dict, config, context)
                yield result

        return SimpleAgentWrapper(chain)

    def _adjust_prompt_based_on_settings(self):
        """根据功能开关调整系统提示词"""
        if self.agent_type == "student":
            # 学生提示词调整
            base_prompt = STUDENT_PROMPT

            if self.parallel_thinking_enabled:
                base_prompt += "\n\n" + PARALLEL_THINKING_PROMPT

            # 根据数学背景调整
            if self.math_background_level == "beginner":
                base_prompt += "\n\n" + MATH_BACKGROUND_BEGINNER
            elif self.math_background_level == "advanced":
                base_prompt += "\n\n" + MATH_BACKGROUND_ADVANCED
            else:
                base_prompt += "\n\n" + MATH_BACKGROUND_INTERMEDIATE

            self.prompt_sys = base_prompt

        elif self.agent_type == "teacher":
            # 教师提示词调整 - 包含正确答案知识但强调不要泄露
            base_prompt = TEACHER_PROMPT

            if self.correct_answer:
                base_prompt += f"\n\nYou know the correct answer is: {self.correct_answer}. But DO NOT reveal this answer directly to the student. Guide them to discover it themselves."

            if self.socratic_teaching_enabled:
                base_prompt += "\n\n" + SOCRATIC_TEACHING_PROMPT

            self.prompt_sys = base_prompt

    def config_create(self, key_i, value_i):
        self.agent_config = {"configurable": {key_i: value_i}, "recursion_limit": 100}

    def context_set(self, **kwargs):
        self.context = self.context_schema(**kwargs)

    def chat_once(self, user_input, response_type='invoke', silence=False, **kwargs):
        if response_type == 'invoke':
            self.agent_response_invoke(user_input, **kwargs)
            if not silence:
                self.agent_output()
        else:
            self.agent_response_stream(user_input, **kwargs)

    def multi_agent_chat_explicit(self, teacher_agent, problem: str, correct_answer: str,
                                  dialogue_record: DialogueRecord, **kwargs):
        """模式1: 显式交互 - 教师和学生直接对话"""
        if dialogue_record.debug_mode:
            logger.info(f"\n🎯 开始解题: {problem}")
            logger.info("=" * 50)

        # 设置教师知道的正确答案
        teacher_agent.set_correct_answer(correct_answer)
        teacher_agent._adjust_prompt_based_on_settings()  # 重新调整提示词

        # 重置对话历史
        self.dialogue_history = []
        self.current_turn = 0

        # 学生首次尝试
        student_response = self._invoke_agent(problem)
        self.dialogue_history.append(("student", student_response))

        # 分析学生回复
        parallel_count, path_count = dialogue_record.analyze_student_response(student_response)

        # 记录第一轮对话
        dialogue_record.add_turn({
            "turn": 1,
            "student_response": student_response,
            "teacher_response": "",
            "teacher_intent": "initial_response",
            "parallel_thinking_count": parallel_count,
            "thinking_paths_count": path_count,
            "answer_leakage": False
        })

        if dialogue_record.debug_mode:
            logger.info(f"👨‍🎓 学生 [轮次1]: {student_response}")

        self.correct_answer = extract_answer(correct_answer)
        dialogue_record.first_correct = self._has_correct_answer(extract_answer(student_response), self.correct_answer)

        for turn in range(self.max_turns):

            # 检查学生是否得出正确答案
            self.student_answer = extract_answer(student_response)
            if dialogue_record.first_correct or self._has_correct_answer(self.student_answer, self.correct_answer):
                if dialogue_record.debug_mode:
                    logger.info("🎉 学生得出正确答案!")
                dialogue_record.correct = True
                dialogue_record.final_student_answer = self.student_answer
                return self.student_answer, self.correct_answer, dialogue_record

            current_turn = turn + 2  # 从第二轮开始

            if dialogue_record.debug_mode:
                logger.info(f"\n🔄 第 {current_turn} 轮对话:")
                logger.info("-" * 30)

            # 教师回应
            teacher_input = self._format_teacher_input(problem, self.dialogue_history)
            if dialogue_record.debug_mode:
                logger.info(f"👨‍🏫 teacher_input [轮次{current_turn}]: {teacher_input}")
            teacher_response = teacher_agent._invoke_agent(teacher_input)
            self.dialogue_history.append(("teacher", teacher_response))

            # 检查答案泄露
            leakage_detected = dialogue_record.check_answer_leakage(teacher_response)

            # 检查是否应该结束对话
            if self._should_end_dialogue(teacher_response):
                if dialogue_record.debug_mode:
                    logger.info("✅ 教师认为解题完成")
                dialogue_record.final_student_answer = self.student_answer
                return self.student_answer, self.correct_answer, dialogue_record

            if dialogue_record.debug_mode:
                logger.info(f"👨‍🏫 教师 [轮次{current_turn}]: {teacher_response}")
            if leakage_detected:
                if dialogue_record.debug_mode:
                    logger.info("⚠️  检测到答案泄露!")

            # 学生回应
            student_input = self._format_student_input(problem, self.dialogue_history)
            if dialogue_record.debug_mode:
                logger.info(f"👨‍🎓 student_input [轮次{current_turn}]: {student_input}")
            student_response = self._invoke_agent(student_input)
            self.dialogue_history.append(("student", student_response))

            # 分析学生回复
            parallel_count, path_count = dialogue_record.analyze_student_response(student_response)

            # 记录本轮对话
            dialogue_record.add_turn({
                "turn": current_turn,
                "student_response": student_response,
                "teacher_response": teacher_response,
                "teacher_intent": self._analyze_teacher_intent(teacher_response),
                "parallel_thinking_count": parallel_count,
                "thinking_paths_count": path_count,
                "answer_leakage": leakage_detected
            })

            if dialogue_record.debug_mode:
                logger.info(f"👨‍🎓 学生 [轮次{current_turn}]: {student_response}")

            self.current_turn = current_turn

        # 设置最终答案
        dialogue_record.final_student_answer = self.student_answer

        return self._get_final_answer(), self.correct_answer, dialogue_record

    def _analyze_teacher_intent(self, teacher_response: str) -> str:
        """分析教师回复的意图"""
        response_lower = teacher_response.lower()

        if any(word in response_lower for word in ["question", "ask", "what do you think"]):
            return "socratic_questioning"
        elif any(word in response_lower for word in ["hint", "suggest", "try"]):
            return "providing_hint"
        elif any(word in response_lower for word in ["correct", "right", "good"]):
            return "positive_feedback"
        elif any(word in response_lower for word in ["wrong", "incorrect", "mistake"]):
            return "correcting_error"
        elif any(word in response_lower for word in ["explain", "concept", "principle"]):
            return "explaining_concept"
        else:
            return "general_guidance"

    def multi_agent_chat_tool_based(self, problem: str, correct_answer: str,
                                    dialogue_record: DialogueRecord, **kwargs):
        """模式2: 工具调用 - 学生作为controller调用教师工具"""
        print(f"\n🎯 开始工具调用模式解题: {problem}")
        print("=" * 50)

        # 配置学生agent以包含教师工具（包含正确答案）
        teacher_tool = self._create_teacher_tool(correct_answer)
        self.tools.append(teacher_tool)

        # 重新初始化agent以包含新工具
        self.agent = create_agent(
            model=self.model,
            system_prompt=self.prompt_sys + "\n\nYou can use the ask_teacher tool when you need guidance.",
            tools=self.tools,
            context_schema=self.context_schema,
            response_format=self.response_format,
            checkpointer=self.checkpointer,
            middleware=self.middleware_list,
        )

        # 学生自主解题，可在需要时调用教师工具
        final_response = self._invoke_agent(problem)

        # 分析学生回复
        parallel_count, path_count = dialogue_record.analyze_student_response(final_response)

        # 记录工具调用模式的对话（简化为单轮）
        dialogue_record.add_turn({
            "turn": 1,
            "student_response": final_response,
            "teacher_response": "Tool-based interaction",
            "teacher_intent": "tool_guidance",
            "parallel_thinking_count": parallel_count,
            "thinking_paths_count": path_count,
            "answer_leakage": False
        })

        dialogue_record.final_student_answer = final_response

        print(f"👨‍🎓 学生最终回答: {final_response}")

        return final_response

    def _create_teacher_tool(self, correct_answer: str):
        """创建教师工具供学生调用（包含正确答案知识）"""
        from langchain.tools import tool

        @tool
        def ask_teacher(question: str) -> str:
            """Ask the teacher for guidance on a specific question or problem.

            The teacher knows the correct answer but will not reveal it directly.
            Instead, the teacher will provide helpful guidance and hints.

            Use this tool when:
            - You're stuck on a math problem
            - You need clarification on concepts
            - You want to check your approach
            - You need step-by-step guidance
            """
            # 基于正确答案提供引导性提示
            guidance_responses = [
                "Let me guide you through this step by step. What part are you finding difficult?",
                "Good attempt! Let's break this down. What's your current approach?",
                "I see where you might be confused. Let me ask you a question to help you think differently...",
                "Remember the key concept here is to identify the known values and what you're solving for.",
                "Try breaking the problem into smaller parts. What's the first step you would take?",
                "Consider what information you have and what you're trying to find. How can you connect them?",
                "That's a good start. Now think about what mathematical operations might be needed here."
            ]
            import random
            return random.choice(guidance_responses)

        return ask_teacher

    def _invoke_agent(self, input_text):
        """调用agent并返回响应"""
        response = self.agent.invoke({
            "messages": [{"role": "user", "content": input_text}]
        }, config=self.agent_config, context=self.context)

        if isinstance(response, str):
            return response
        if response and 'messages' in response:
            return response['structured_response'].main_response
        return ""

    def _format_teacher_input(self, problem, history):
        """格式化教师输入"""
        context = f"Problem: {problem}\n\nDialogue History:\n"
        for role, content in history:
            context += f"{role}: {content}\n\n"
        context += 'System: '
        # 根据苏格拉底教学开关调整提示
        if self.socratic_teaching_enabled:
            context += "Please use Socratic questioning to guide the student. Ask thought-provoking questions rather than giving direct answers."
        else:
            context += "Please provide appropriate guidance to help the student solve the problem."

        return context

    def _format_student_input(self, problem, history):
        """格式化学生输入"""
        context = f"Original Problem: {problem}\n\nDialogue History:\n"
        for role, content in history:
            context += f"{role}: {content}\n\n"
        context += 'System: '
        # 根据并行思考开关调整提示
        if self.parallel_thinking_enabled:
            context += "Use parallel thinking to consider multiple approaches to this problem. Use <Parallel> and </Parallel> tags for parallel thinking, and <Path> </Path> tags for different thinking paths."

        context += "Please continue solving the problem or respond to the teacher's guidance:"
        return context

    def _should_end_dialogue(self, teacher_response):
        """判断是否应该结束对话"""
        end_phrases = ["finished", "completed"]
        return any(phrase in teacher_response.lower() for phrase in end_phrases)

    def _has_correct_answer(self, student_answer, correct_answer):
        """判断学生是否得出正确答案"""
        # 使用现有的答案提取和比较逻辑
        return is_equiv_MATH(correct_answer, student_answer)

    def _get_final_answer(self):
        """获取最终答案"""
        student_final_answer = ""
        if self.dialogue_history:
            for role, content in self.dialogue_history:
                if role == "student":
                    student_final_answer = content[1]
            return student_final_answer
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

    def agent_output(self, all_messages=False):
        if all_messages:
            for i in self.response['messages']:
                print(f"{type(i)}: {i.content}")
        else:
            print(f"{type(self.response['messages'][-1])}: {self.response['structured_response'].main_response}")

    def get_dialogue_summary(self):
        """获取对话摘要"""
        summary = f"Dialogue Summary ({self.current_turn} turns):\n"
        for i, (role, content) in enumerate(self.dialogue_history):
            summary += f"Turn {i + 1} ({role}): {content[:100]}...\n"
        return summary


# 测试代码
def test_dialogue_record():
    """测试对话记录功能"""
    print("🧪 测试对话记录功能...")

    record = DialogueRecord("What is 2+2?", "4")
    record.add_turn({
        "turn": 1,
        "student_response": "I think it's <Parallel>maybe 3</Parallel> or <Path>maybe 4</Path>",
        "teacher_response": "Think about basic addition",
        "teacher_intent": "guidance",
        "parallel_thinking_count": 1,
        "thinking_paths_count": 1,
        "answer_leakage": False
    })

    record.analyze_student_response("Another <Parallel>thought</Parallel>")
    record.check_answer_leakage("The answer is 4")

    assert record.parallel_thinking_count == 2
    assert record.thinking_paths_count == 1
    assert record.leaked_answer == True
    assert record.total_turns == 1

    print("✅ 对话记录测试通过!")


def test_experiment_recorder():
    """测试实验记录器"""
    print("🧪 测试实验记录器...")

    recorder = ExperimentRecorder("test_experiment")

    record1 = DialogueRecord("Problem 1", "Answer 1")
    record1.correct = True
    record1.parallel_thinking_count = 2
    record1.thinking_paths_count = 3
    record1.total_turns = 2

    record2 = DialogueRecord("Problem 2", "Answer 2")
    record2.correct = False
    record2.parallel_thinking_count = 1
    record2.thinking_paths_count = 2
    record2.total_turns = 3

    recorder.add_record(record1)
    recorder.add_record(record2)

    stats = recorder.calculate_statistics()

    assert stats["total_problems"] == 2
    assert stats["accuracy"] == 0.5
    assert stats["avg_turns_per_problem"] == 2.5
    assert stats["avg_parallel_thinking"] == 1.5
    assert stats["avg_thinking_paths"] == 2.5

    print("✅ 实验记录器测试通过!")


if __name__ == "__main__":
    # 运行测试
    test_dialogue_record()
    test_experiment_recorder()
    print("🎉 所有测试通过!")


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