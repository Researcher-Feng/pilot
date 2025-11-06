# agent_IO.py
import logging
import os
import sys
sys.path.append(r'../..')
from typing import Callable, List, Optional, Dict, Any
from types import SimpleNamespace
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


class LocalModelWrapper:
    """本地模型包装器，模拟API模型的功能"""

    def __init__(self, model_config):
        self.model_config = model_config
        self.model = ChatOllama(
            model=model_config.model_name,
            base_url=model_config.base_url,
            temperature=model_config.temperature,
            num_predict=model_config.max_tokens,
            **model_config.extra_params
        )
        self.model_name = model_config.model_name

    def invoke(self, messages):
        """模拟API调用"""
        if isinstance(messages, dict) and 'messages' in messages:
            # 处理LangChain格式的输入
            input_text = self._extract_user_input(messages['messages'])
        else:
            input_text = str(messages)

        response = self.model.invoke(input_text)
        return self._format_response(response)

    def stream(self, messages):
        """模拟流式响应"""
        input_text = self._extract_user_input(messages['messages'])
        response = self.model.invoke(input_text)
        yield self._format_response(response)

    def _extract_user_input(self, messages):
        """从消息列表中提取用户输入"""
        for msg in reversed(messages):
            if hasattr(msg, 'content'):
                return msg.content
            elif isinstance(msg, dict) and 'content' in msg:
                return msg['content']
        return ""

    def _format_response(self, response):
        """格式化响应以兼容API格式"""
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)

        return {
            'messages': [{'role': 'assistant', 'content': content}],
            'structured_response': SimpleNamespace(main_response=content)
        }


class SimpleLocalAgent:
    """简化版本地Agent，支持核心功能"""

    def __init__(self, agent_type="student", debug_mode=False):
        self.agent_type = agent_type
        self.model = None
        self.system_prompt = ""
        self.tools = []
        self.context = {}
        self.dialogue_history = []
        self.debug_mode = debug_mode
        self.last_full_prompt = ""  # 记录最后一次完整prompt

    def init_model(self, model_config):
        """初始化本地模型"""
        self.model = LocalModelWrapper(model_config)

    def set_system_prompt(self, prompt):
        """设置系统提示词"""
        self.system_prompt = prompt

    def add_tools(self, tools_list):
        """添加工具（简化实现）"""
        self.tools = tools_list

    def set_context(self, **kwargs):
        """设置上下文"""
        self.context.update(kwargs)

    def invoke(self, input_dict, config=None, context=None):
        """调用Agent"""
        user_input = self._extract_input(input_dict)

        # 构建完整提示词
        full_prompt = self._build_full_prompt(user_input)
        self.last_full_prompt = full_prompt  # 保存完整prompt

        # Debug输出
        if self.debug_mode:
            logger.info("\n" + "=" * 80)
            logger.info(f"🔍 DEBUG - {self.agent_type.upper()} AGENT FULL PROMPT:")
            logger.info("=" * 80)
            logger.info(full_prompt)
            logger.info("=" * 80 + "\n")

        # 调用模型
        response = self.model.invoke(full_prompt)

        # 记录对话历史
        self.dialogue_history.append(("user", user_input))
        self.dialogue_history.append(("assistant", response['structured_response'].main_response))

        return response

    def get_last_prompt(self):
        """获取最后一次的完整prompt"""
        return self.last_full_prompt

    def _extract_input(self, input_dict):
        """提取输入文本"""
        messages = input_dict.get('messages', [])
        for msg in reversed(messages):
            if hasattr(msg, 'content'):
                return msg.content
            elif isinstance(msg, dict) and 'content' in msg:
                return msg['content']
        return ""

    def _build_full_prompt(self, user_input):
        """构建完整提示词"""
        prompt_parts = []

        # 系统提示词
        if self.system_prompt:
            prompt_parts.append(f"System: {self.system_prompt}")

        # 上下文信息
        if self.context:
            context_str = ", ".join([f"{k}: {v}" for k, v in self.context.items()])
            prompt_parts.append(f"Context: {context_str}")

        # 对话历史
        if self.dialogue_history:
            history_str = "\n".join([f"{role}: {content}" for role, content in self.dialogue_history[-6:]])  # 最近3轮对话
            prompt_parts.append(f"Dialogue History:\n{history_str}")

        # 当前输入
        prompt_parts.append(f"Teacher: {user_input}")
        prompt_parts.append("Assistant:")

        return "\n\n".join(prompt_parts)


class SmartSummaryMemory:
    """智能对话摘要内存管理"""

    def __init__(self, llm=None, max_turns=10, max_token_limit=2000, enabled=False, debug_mode=False):
        self.debug_mode = debug_mode
        self.enabled = enabled
        self.max_turns = max_turns
        self.max_token_limit = max_token_limit
        self.llm = llm
        self.conversation_history = []
        self.summary_history = []  # 存储历史摘要
        self.turn_count = 0
        self.current_summary = ""

    def add_message(self, role: str, content: str):
        """添加消息到对话历史"""
        if not self.enabled:
            return

        self.conversation_history.append({"role": role, "content": content})
        self.turn_count += 1

        # 检查是否需要生成摘要
        if self._should_generate_summary():
            self._generate_summary()

    def _should_generate_summary(self):
        """判断是否需要生成摘要"""
        if not self.enabled:
            return False

        # 基于轮次判断
        if self.turn_count >= self.max_turns:
            return True

        # 基于token数量判断（简单估算）
        total_chars = sum(len(msg["content"]) for msg in self.conversation_history)
        estimated_tokens = total_chars // 4  # 简单估算：1 token ≈ 4 characters
        if estimated_tokens > self.max_token_limit:
            return True

        return False

    def _generate_summary(self):
        """生成对话摘要"""
        if not self.enabled or not self.llm:
            return

        try:
            # 构建摘要提示词
            conversation_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in self.conversation_history]
            )

            summary_prompt = f"""Please summarize the key information from this math tutoring conversation, focusing on:

1. The main math problem being discussed
2. Student's current approach and difficulties
3. Teacher's guidance and key questions asked
4. Important intermediate steps and concepts
5. Current progress toward solution

Conversation:
{conversation_text}

Provide a concise but informative summary in English:"""

            # 调用LLM生成摘要
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(summary_prompt)
                summary = response.content if hasattr(response, 'content') else str(response)
            else:
                # 简化调用
                summary = "Summary: Math problem discussion in progress"

            self.current_summary = summary
            self.summary_history.append({
                "turn_count": self.turn_count,
                "summary": summary,
                "timestamp": datetime.datetime.now().isoformat()
            })

            # 保留最近对话作为上下文
            keep_recent = min(3, len(self.conversation_history))
            self.conversation_history = self.conversation_history[-keep_recent:]
            self.turn_count = keep_recent

            if self.debug_mode:
                logger.info(f"📝 已生成对话摘要 (第{len(self.summary_history)}次摘要)")
                logger.info(f"Summary: {summary[:200]}...")

        except Exception as e:
            logger.warning(f"❌ 生成摘要时出错: {e}")
            self.current_summary = f"Conversation history: {len(self.conversation_history)} turns about math problem"

    def get_context(self):
        """获取当前上下文（包含摘要）"""
        if not self.enabled:
            return ""

        context_parts = []

        # 添加当前摘要
        if self.current_summary:
            context_parts.append(f"Previous Conversation Summary:\n{self.current_summary}")

        # 添加最近对话历史
        recent_messages = self.conversation_history[-3:]  # 保留最近3轮
        if recent_messages:
            recent_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in recent_messages]
            )
            context_parts.append(f"Recent Dialogue:\n{recent_text}")

        return "\n\n".join(context_parts) if context_parts else ""

    def clear(self):
        """清空内存"""
        self.conversation_history.clear()
        self.summary_history.clear()
        self.turn_count = 0
        self.current_summary = ""

    def get_stats(self):
        """获取内存统计信息"""
        return {
            "enabled": self.enabled,
            "total_turns": self.turn_count,
            "conversation_history_count": len(self.conversation_history),
            "summary_history_count": len(self.summary_history),
            "current_summary_length": len(self.current_summary)
        }


class SummaryConfig:
    """摘要配置类"""
    def __init__(self, enabled=False, max_turns=10, max_token_limit=2000, summary_model=None):
        self.enabled = enabled
        self.max_turns = max_turns
        self.max_token_limit = max_token_limit
        self.summary_model = summary_model  # 专门用于摘要的模型配置


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
        parallel_count = response.count('<Parallel')
        self.parallel_thinking_count += parallel_count

        # 统计思考路径标签
        path_count = response.count('<Path')
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



class StudentCognitiveState:
    """学生认知状态管理"""

    def __init__(self,
                 carelessness_level=5,
                 math_background="intermediate",
                 response_style="thoughtful",
                 preferred_method="balanced",
                 learning_style="visual"):

        self.carelessness_level = carelessness_level  # 1-10
        self.math_background = math_background  # beginner/intermediate/advanced
        self.response_style = response_style  # thoughtful/impulsive/detailed/brief
        self.preferred_method = preferred_method  # algebraic/geometric/computational/balanced
        self.learning_style = learning_style  # visual/auditory/kinesthetic/reading-writing

        # 动态属性
        self.problem_solving_history = []
        self.error_patterns = []
        self.method_preferences = []
        self.conceptual_understanding = {}

    def update_based_on_interaction(self, problem, student_approach, errors, method_used, success):
        """基于交互更新认知状态"""
        # 记录问题解决历史
        self.problem_solving_history.append({
            "problem": problem,
            "approach": student_approach,
            "errors": errors,
            "method_used": method_used,
            "success": success,
            "timestamp": datetime.datetime.now().isoformat()
        })

        # 更新粗心程度（基于错误模式）
        if errors:
            self.error_patterns.extend(errors)
            # 如果频繁出现计算错误，增加粗心程度
            calculation_errors = sum(1 for error in errors if "calculation" in error.lower())
            if calculation_errors > 0:
                self.carelessness_level = min(10, self.carelessness_level + 0.5)

        # 更新方法偏好
        self.method_preferences.append(method_used)

        # 更新数学背景（基于成功率和方法复杂度）
        if success and len(student_approach) > 3:  # 复杂的成功解法
            if self.math_background == "beginner":
                self.math_background = "intermediate"
            elif self.math_background == "intermediate" and len(self.problem_solving_history) > 5:
                self.math_background = "advanced"

    def to_dict(self):
        """转换为字典"""
        return {
            "carelessness_level": self.carelessness_level,
            "math_background": self.math_background,
            "response_style": self.response_style,
            "preferred_method": self.preferred_method,
            "learning_style": self.learning_style,
            "problem_solving_count": len(self.problem_solving_history),
            "recent_success_rate": self._calculate_recent_success_rate()
        }

    def _calculate_recent_success_rate(self):
        """计算最近的成功率"""
        if len(self.problem_solving_history) == 0:
            return 0.0
        recent = self.problem_solving_history[-5:]  # 最近5个问题
        successes = sum(1 for record in recent if record["success"])
        return successes / len(recent)

    def get_prompt_context(self):
        """获取提示词上下文"""
        return STUDENT_COGNITIVE_STATE_PROMPT.format(
            carelessness_level=self.carelessness_level,
            math_background=self.math_background,
            response_style=self.response_style,
            preferred_method=self.preferred_method,
            learning_style=self.learning_style
        )


class SolutionTree:
    """解题树管理"""

    def __init__(self, problem_statement):
        self.problem_statement = problem_statement
        self.root_node = {"type": "problem", "content": problem_statement}
        self.solution_paths = []
        self.current_student_path = []

    def add_expert_path(self, path_data):
        """添加专家解题路径"""
        self.solution_paths.append({
            "type": "expert",
            "path_id": len(self.solution_paths) + 1,
            "method": path_data.get("method", "unknown"),
            "complexity": path_data.get("complexity", "medium"),
            "innovation": path_data.get("innovation", "medium"),
            "steps": path_data.get("steps", []),
            "intermediate_answers": path_data.get("intermediate_answers", []),
            "final_answer": path_data.get("final_answer", "")
        })

    def add_student_step(self, step_content, method_used=None):
        """添加学生解题步骤"""
        # 如果没有提供方法，尝试从步骤内容中检测
        if method_used is None:
            method_used = self._detect_student_method(step_content)

        self.current_student_path.append({
            "step_number": len(self.current_student_path) + 1,
            "content": step_content,
            "method": method_used,
            "timestamp": datetime.datetime.now().isoformat()
        })

    def complete_student_path(self, success, final_answer=None):
        """完成学生解题路径"""
        student_path = {
            "type": "student",
            "path_id": f"student_{len(self.solution_paths) + 1}",
            "method": self._detect_student_method(),
            "steps": self.current_student_path.copy(),
            "success": success,
            "final_answer": final_answer
        }
        self.solution_paths.append(student_path)
        self.current_student_path = []

        return student_path

    def _detect_student_method(self, response=None):
        """检测学生使用的方法
        Args:
            response: 可选的响应文本，如果不提供则使用当前学生路径
        """
        if response:
            # 如果提供了响应文本，直接分析该文本
            content = response.lower()
        elif self.current_student_path:
            # 否则使用当前学生路径的所有内容
            content = " ".join([step["content"].lower() for step in self.current_student_path])
        else:
            return "unknown"

        # if not self.current_student_path:
        #     return "unknown"
        #
        # content = " ".join([step["content"].lower() for step in self.current_student_path])

        if any(word in content for word in ["equation", "solve for", "variable", "x ="]):
            return "algebraic"
        elif any(word in content for word in ["diagram", "graph", "shape", "angle", "area"]):
            return "geometric"
        elif any(word in content for word in ["calculate", "compute", "number", "digit"]):
            return "computational"
        elif any(word in content for word in ["logic", "reason", "therefore", "because", "since", "if then"]):
            return "logical"
        elif any(word in content for word in ["guess", "try", "maybe", "perhaps", "i think"]):
            return "trial_and_error"
        else:
            return "unknown"

    def compare_with_expert(self):
        """比较学生路径与专家路径"""
        if not self.current_student_path:
            return {"similarity": 0, "closest_expert_path": None}

        student_method = self._detect_student_method()
        expert_paths = [p for p in self.solution_paths if p["type"] == "expert"]

        if not expert_paths:
            return {"similarity": 0, "closest_expert_path": None}

        # 简单的方法匹配
        matching_paths = [p for p in expert_paths if p["method"] == student_method]
        if matching_paths:
            closest = matching_paths[0]
            similarity = 0.8  # 方法匹配的基础相似度
        else:
            closest = expert_paths[0]
            similarity = 0.3  # 方法不匹配的基础相似度

        return {
            "similarity": similarity,
            "closest_expert_path": closest,
            "method_match": student_method == closest["method"]
        }

    def to_dict(self):
        """转换为字典"""
        return {
            "problem_statement": self.problem_statement,
            "solution_paths": self.solution_paths,
            "current_student_path": self.current_student_path
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

    def __init__(self, agent_type: str = "student", debug_mode: bool = False):
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
        self.math_background_level = False
        self.debug_mode = debug_mode

        # Memory 与 Summary
        self.local_agent = None
        self.memory = None  # 添加内存管理
        self.summary_config = None
        self.summary_llm = None  # 专门的摘要LLM

        # 解题树
        self.cognitive_state = None  # 学生认知状态
        self.solution_tree = None    # 当前解题树
        self.use_cognitive_state = False
        self.use_solution_tree = False

    def set_cognitive_state(self, cognitive_state: StudentCognitiveState):
        """设置认知状态"""
        self.cognitive_state = cognitive_state
        self.use_cognitive_state = True

    def set_solution_tree(self, solution_tree: SolutionTree):
        """设置解题树"""
        self.solution_tree = solution_tree
        self.use_solution_tree = True

    def _build_full_prompt(self, user_input, memory_context):
        """构建完整提示词（增强版）"""
        prompt_parts = []

        # 系统提示词
        if self.prompt_sys:
            # 处理字符串类型的系统提示词
            if isinstance(self.prompt_sys, str):
                # 如果是预定义的提示词变量名，尝试获取其值
                if self.prompt_sys in globals():
                    system_prompt = globals()[self.prompt_sys]
                else:
                    system_prompt = self.prompt_sys
            else:
                system_prompt = str(self.prompt_sys)
            prompt_parts.append(f"System: {system_prompt}")

        # 认知状态（如果是学生且启用）
        if (self.agent_type == "student" and self.use_cognitive_state and
                self.cognitive_state and hasattr(self.cognitive_state, 'get_prompt_context')):
            cognitive_context = self.cognitive_state.get_prompt_context()
            prompt_parts.append(f"Student Profile: {cognitive_context}")

        # 解题树上下文（如果启用）
        if self.use_solution_tree and self.solution_tree:
            try:
                tree_context = self._build_solution_tree_context()
                if tree_context:
                    prompt_parts.append(f"Solution Context: {tree_context}")
            except Exception as e:
                if self.debug_mode:
                    logger.info(f"❌ 解题树上下文构建失败: {e}")

        # 内存上下文
        if memory_context:
            prompt_parts.append(f"Conversation Context: {memory_context}")

        # 对话历史（如果有）
        if hasattr(self, 'dialogue_history') and self.dialogue_history:
            history_str = "\n".join([f"{role}: {content}" for role, content in self.dialogue_history[-4:]])
            prompt_parts.append(f"Recent Dialogue:\n{history_str}")

        # 当前输入
        prompt_parts.append(f"Current Input: {user_input}")
        prompt_parts.append("Please respond:")

        return "\n\n".join(prompt_parts)

    def _build_solution_tree_context(self):
        """构建解题树上下文"""
        if not self.solution_tree:
            return ""

        context_parts = []

        try:
            if self.agent_type == "teacher":
                try:
                    # 教师看到专家路径和学生路径的比较
                    if hasattr(self.solution_tree, 'compare_with_expert'):
                        comparison = self.solution_tree.compare_with_expert()
                        if comparison and comparison.get("closest_expert_path"):
                            expert_method = comparison['closest_expert_path'].get('method', 'unknown')
                            similarity = comparison.get('similarity', 0)
                            context_parts.append(
                                f"Expert solution available using {expert_method} method")
                            context_parts.append(f"Similarity to student approach: {similarity:.2f}")
                except Exception as e:
                    if self.debug_mode:
                        print(f"⚠️ 解题树比较失败: {e}")
                    context_parts.append("Expert guidance available for this problem")

            elif self.agent_type == "student":
                # 学生看到自己的进度
                if (hasattr(self.solution_tree, 'current_student_path') and
                    self.solution_tree.current_student_path):
                    context_parts.append(
                        f"Your current solution path has {len(self.solution_tree.current_student_path)} steps")
                    context_parts.append("Continue your approach or try a different method if stuck")

                # 添加可用的专家路径信息
            if (hasattr(self.solution_tree, 'solution_paths') and
                    self.solution_tree.solution_paths):
                expert_paths = [p for p in self.solution_tree.solution_paths
                                if hasattr(p, 'get') and p.get("type") == "expert"]
                if expert_paths:
                    methods = set(p.get("method", "unknown") for p in expert_paths)
                    context_parts.append(f"Available expert methods: {', '.join(methods)}")

        except Exception as e:
            if self.debug_mode:
                logger.info(f"❌ 解题树上下文构建异常: {e}")
            # 提供基本的上下文信息
            context_parts.append("Solution guidance is available for this problem")

        return "\n".join(context_parts) if context_parts else ""

    def _parse_solution_tree_student(self, response, problem):
        """解析学生的解题树响应"""
        solution_tree = SolutionTree(problem)

        try:
            # 简单的解析逻辑 - 在实际应用中可以使用更复杂的解析
            if "<SolutionTree>" in response and self.agent_type == "student":
                # 提取解决方案路径
                paths_section = response.split("<SolutionPaths>")[1].split("</SolutionPaths>")[0]
                path_blocks = paths_section.split("</Path>")

                for block in path_blocks:
                    if "<Path" in block:
                        # 提取路径信息
                        method = self._extract_site_tag(block, "method")
                        # 提取步骤
                        steps = []
                        intermediate_answers = []
                        step_parts = block.split("<Step")
                        for step_part in step_parts[1:]:
                            if ">" in step_part and "</Step>" in step_part:
                                step_content = step_part.split(">", 1)[1].split("</Step>")[0]
                                steps.append(step_content)
                            if "<IntermediateAnswer>" in step_part and "</IntermediateAnswer>" in step_part:
                                intermediate_content = step_part.split("<IntermediateAnswer>", 1)[1].split("</IntermediateAnswer>")[0]
                                intermediate_answers.append(intermediate_content)

                        # 提取最终答案
                        final_answer = self._extract_xml_tag(block, "FinalAnswer")

                        solution_tree.add_expert_path({
                            "method": method,
                            "steps": steps,
                            "intermediate_answers": intermediate_answers,
                            "final_answer": final_answer
                        })

        except Exception as e:
            print(f"❌ Error parsing solution tree: {e}")
            # 如果解析失败，创建一个默认的解决方案路径
            solution_tree.add_expert_path({
                "method": "algebraic",
                "steps": ["Apply standard algebraic approach", "Solve step by step"],
                "final_answer": "[[Answer will be determined]]"
            })

        return solution_tree

    def _extract_xml_tag(self, text, tag_name):
        """提取XML标签内容"""
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"

        if start_tag in text and end_tag in text:
            return text.split(start_tag)[1].split(end_tag)[0].strip()
        return ""

    def _extract_site_tag(self, text, tag_name):
        """提取XML标签内容"""
        start_tag = f'{tag_name}="'
        end_tag = f'"'

        if start_tag in text and end_tag in text:
            return text.split(start_tag)[1].split(end_tag)[0].strip()
        return ""

    def record_student_step(self, step_content, method_used=None):
        """记录学生解题步骤"""
        if (self.agent_type == "student" and self.use_solution_tree and
                self.solution_tree and hasattr(self.solution_tree, 'add_student_step')):
            try:
                self.solution_tree.add_student_step(step_content, method_used)

                # 添加调试信息
                if hasattr(self, 'debug_mode') and self.debug_mode:
                    detected_method = method_used if method_used else self.solution_tree._detect_student_method(
                        step_content)
                    print(f"📝 记录解题步骤: 方法={detected_method}, 内容长度={len(step_content)}")
            except Exception as e:
                if self.debug_mode:
                    print(f"❌ 记录解题步骤失败: {e}")


    def complete_student_solution(self, success, final_answer=None):
        """完成学生解题"""
        if (self.agent_type == "student" and self.use_solution_tree and
                self.solution_tree and hasattr(self.solution_tree, 'complete_student_path')):
            result = self.solution_tree.complete_student_path(success, final_answer)

            # 添加调试信息
            if hasattr(self, 'debug_mode') and self.debug_mode:
                print(f"🌳 解题树完成: 成功={success}, 答案={final_answer}")
                if hasattr(self.solution_tree, 'current_student_path'):
                    print(f"   路径步骤数: {len(self.solution_tree.current_student_path)}")
                if hasattr(self.solution_tree, 'solution_paths'):
                    print(f"   总解决方案路径: {len(self.solution_tree.solution_paths)}")

            return result
        return None

    def enable_conversation_summary(self, summary_config: SummaryConfig, summary_llm=None):
        """启用对话摘要功能"""
        self.summary_config = summary_config
        self.summary_llm = summary_llm

        self.memory = SmartSummaryMemory(
            llm=summary_llm,
            max_turns=summary_config.max_turns,
            max_token_limit=summary_config.max_token_limit,
            enabled=summary_config.enabled
        )

        if hasattr(self.memory, 'debug_mode'):
            self.memory.debug_mode = hasattr(self, 'debug_mode') and self.debug_mode

        status = "enabled" if summary_config.enabled else "disabled"
        logger.info(f"✅ Conversation summary {status} for {self.agent_type} agent")

    def model_init(self, model_config: ModelConfig, model_name=None):
        """初始化模型，支持API和本地调用"""
        if model_config.model_type == "local":
            # 使用简化的本地Agent
            self.local_agent = SimpleLocalAgent(self.agent_type, self.debug_mode)
            self.local_agent.init_model(model_config)
            print(f"✅ 初始化 {model_name} 本地模型: {model_config.model_name}")
        else:
            # API模型配置
            api_kwargs = {
                "temperature": model_config.temperature,
                "timeout": model_config.timeout,
                "max_tokens": model_config.max_tokens,
            }
            self.model = init_chat_model(model_config.model_name, **api_kwargs)

            # 为API模型添加debug中间件
            if self.debug_mode:
                self._add_debug_middleware()

            print(f"✅ 初始化 {model_name} API模型: {model_config.model_name}")
            self.middleware_er = MiddlewareFunc(self.model, self.model)

    def _add_debug_middleware(self):
        """为API模型添加debug中间件"""

        @wrap_model_call
        def debug_middleware(request: ModelRequest, handler) -> ModelResponse:
            print("\n" + "=" * 80)
            print(f"🔍 DEBUG - {self.agent_type.upper()} AGENT API REQUEST:")
            print("=" * 80)

            # 打印系统提示词
            if hasattr(request, 'system_prompt') and request.system_prompt:
                print("System Prompt:")
                print(request.system_prompt)
                print("-" * 40)

            # 打印消息历史
            messages = request.state.get("messages", [])
            print("Message History:")
            for i, msg in enumerate(messages):
                role = getattr(msg, 'role', type(msg).__name__)
                content = getattr(msg, 'content', str(msg))
                print(f"{i}. {role}: {content}")

            print("=" * 80 + "\n")

            return handler(request)

        # 添加到中间件列表
        if not hasattr(self, 'middleware_list'):
            self.middleware_list = []
        self.middleware_list.append(debug_middleware)

    def get_debug_info(self):
        """获取debug信息"""
        if hasattr(self, 'local_agent') and self.local_agent:
            return {
                "agent_type": self.agent_type,
                "last_prompt": self.local_agent.get_last_prompt(),
                "dialogue_history": self.local_agent.dialogue_history
            }
        else:
            return {
                "agent_type": self.agent_type,
                "debug_mode": self.debug_mode
            }

    def set_correct_answer(self, correct_answer: str):
        """设置正确答案（教师agent使用）"""
        self.correct_answer = correct_answer

    def agent_init(self, model_config, prompt_sys_name=None, tools_list=None, context_schema=Context,
                   response_format=ResponseFormat, checkpointer=simple_checkpointer,
                   middleware=None, **kwargs):
        """初始化agent配置"""
        # 在解题树模式下，禁用可能导致冲突的工具
        if self.use_solution_tree:
            # 保留必要的工具，移除可能导致工具调用的复杂工具
            safe_tools = []
            if tools_list:
                for tool in tools_list:
                    tool_name = getattr(tool, 'name', str(tool))
                    # 只保留简单、安全的工具
                    if any(safe in tool_name.lower() for safe in ['format', 'memory', 'context']):
                        safe_tools.append(tool)
            self.tools = safe_tools
        else:
            self.tools = tools_list if tools_list else []

        self.context_schema = context_schema
        self.response_format = response_format
        self.checkpointer = checkpointer

        # 设置功能开关
        self.max_turns = kwargs.get("max_turns", 5)
        self.parallel_thinking_enabled = kwargs.get("parallel_thinking", False)
        self.socratic_teaching_enabled = kwargs.get("socratic_teaching", False)

        self.solution_tree = kwargs.get("prompt_solution_tree", False)
        solution_tree_prompt = kwargs.get("prompt_solution_tree", None)
        if solution_tree_prompt:
            self.prompt_sys = solution_tree_prompt
            self.use_solution_tree = True
        else:
            self.prompt_sys = prompt_sys_name
            self.use_solution_tree = False

        # 根据功能开关调整系统提示词
        self._adjust_prompt_based_on_settings(self.prompt_sys)

        # 在解题树模式下，简化中间件配置
        if middleware and self.use_solution_tree:
            # 移除可能导致工具调用冲突的中间件
            safe_middleware = []
            for m in middleware:
                if m not in ['handle_tool_errors', 'dynamic']:
                    safe_middleware.append(m)
            middleware = safe_middleware

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
            # 本地模型初始化
            self.local_agent.set_system_prompt(self.prompt_sys)
            if self.tools:
                self.local_agent.add_tools(self.tools)
            if self.context:
                self.local_agent.set_context(**self.context.__dict__)
        else:
            # 对于API模型，在解题树模式下使用简化的agent创建方式
            if self.use_solution_tree:
                try:
                    self.agent = self._create_simple_agent()
                    if self.debug_mode:
                        print("✅ 使用简化Agent（解题树模式）")
                except Exception as e:
                    if self.debug_mode:
                        print(f"⚠️ 简化Agent创建失败，回退到标准Agent: {e}")
                    self.agent = create_agent(
                        model=self.model,
                        system_prompt=self.prompt_sys,
                        tools=self.tools,  # 使用过滤后的工具
                        context_schema=self.context_schema,
                        response_format=self.response_format,
                        checkpointer=self.checkpointer,
                        middleware=self.middleware_list,
                    )
            else:
                self.agent = create_agent(
                    model=self.model,
                    system_prompt=self.prompt_sys,
                    tools=self.tools,  # 使用过滤后的工具
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
            def __init__(self, chain, debug_mode=False):
                self.chain = chain
                self.debug_mode = debug_mode

            def invoke(self, input_dict, config=None, context=None):
                try:
                    # 提取用户输入
                    messages = input_dict.get("messages", [])
                    user_input = ""
                    for msg in reversed(messages):
                        if hasattr(msg, 'content'):
                            user_input = msg.content
                            break
                        elif isinstance(msg, dict) and 'content' in msg:
                            user_input = msg['content']
                            break

                    if not user_input:
                        user_input = str(messages)
                    # 调用模型
                    response_content = self.chain.invoke({"input": user_input})

                    # 确保response_content是字符串
                    if hasattr(response_content, 'content'):
                        response_text = response_content.content
                    else:
                        response_text = str(response_content)

                    # 返回与现有代码兼容的格式
                    return {
                        'messages': [{'role': 'assistant', 'content': response_text}],
                        'structured_response': SimpleNamespace(main_response=response_text)
                    }

                except Exception as e:
                    if self.debug_mode:
                        print(f"❌ 简单Agent调用错误: {e}")
                    return {
                        'messages': [{'role': 'assistant', 'content': "抱歉，我遇到了一些技术问题。请重新提问。"}],
                        'structured_response': SimpleNamespace(main_response="抱歉，我遇到了一些技术问题。请重新提问。")
                    }

            def stream(self, input_dict, config=None, context=None, stream_mode="values"):
                # 简单的流式响应实现
                result = self.invoke(input_dict, config, context)
                yield result

        return SimpleAgentWrapper(chain, self.debug_mode)

    def _adjust_prompt_based_on_settings(self, prompt_sys):
        """根据功能开关调整系统提示词"""
        if isinstance(prompt_sys, str):
            # 如果是预定义的提示词变量名，尝试获取其值
            if prompt_sys in globals():
                base_prompt = globals()[prompt_sys]
            else:
                base_prompt = prompt_sys
        else:
            base_prompt = str(prompt_sys)

        if self.agent_type == "student":

            if self.parallel_thinking_enabled:
                base_prompt += "\n\n" + PARALLEL_THINKING_PROMPT

            self.prompt_sys = base_prompt

        elif self.agent_type == "teacher":

            if self.correct_answer:
                base_prompt += f"\n\nYou know the correct solution is: {self.correct_answer}. But DO NOT reveal this answer directly to the student. Guide them to discover it themselves."

            if self.socratic_teaching_enabled:
                base_prompt += "\n\n" + SOCRATIC_TEACHING_PROMPT

            self.prompt_sys = base_prompt

        elif self.agent_type == "expert_student":

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

    def _extract_response_text(self, response):
        """从响应对象中提取文本内容"""
        if isinstance(response, str):
            return response

        if hasattr(response, 'get') and isinstance(response, dict):
            # 字典类型的响应
            if 'structured_response' in response and hasattr(response['structured_response'], 'main_response'):
                return response['structured_response'].main_response
            elif 'messages' in response and response['messages']:
                last_msg = response['messages'][-1]
                if hasattr(last_msg, 'content'):
                    return last_msg.content
                elif isinstance(last_msg, dict) and 'content' in last_msg:
                    return last_msg['content']
            # 尝试直接获取content
            elif 'content' in response:
                return response['content']

        if hasattr(response, 'structured_response') and hasattr(response.structured_response, 'main_response'):
            return response.structured_response.main_response
        elif hasattr(response, 'content'):
            return response.content

        # 最后尝试转换为字符串
        return str(response)

    def multi_agent_chat_explicit(self, teacher_agent, problem: str, raw_problem: str, correct_answer: str,
                                  dialogue_record: DialogueRecord, **kwargs):
        """模式1: 显式交互 - 教师和学生直接对话"""
        if dialogue_record.debug_mode:
            logger.info(f"\n🎯 开始解题: {problem}")
            logger.info("=" * 50)

        # 设置教师知道的正确答案
        teacher_agent.set_correct_answer(correct_answer)
        teacher_agent._adjust_prompt_based_on_settings(teacher_agent.prompt_sys)  # 重新调整提示词

        # 重置对话历史
        self.dialogue_history = []
        self.current_turn = 0

        try:
            # 学生首次尝试
            student_response_obj = self._invoke_agent(problem)
            # 确保正确提取响应文本
            student_response = self._extract_response_text(student_response_obj)
            self.dialogue_history.append(("student", student_response))

            # 记录学生第一步解题步骤
            if self.use_solution_tree and self.solution_tree:
                self.record_student_step(student_response, method_used=self._detect_method_from_response(student_response))
                if dialogue_record.debug_mode:
                    logger.info("📝 已记录学生第一次解题步骤")

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

            # 如果第一次就答对了，直接完成解题
            if dialogue_record.first_correct:
                if self.use_solution_tree and self.solution_tree:
                    self.complete_student_solution(success=True, final_answer=student_response)
                    if dialogue_record.debug_mode:
                        logger.info("✅ 学生第一次就答对了，解题路径已完成")
                dialogue_record.correct = True
                dialogue_record.final_student_answer = student_response
                return student_response, self.correct_answer, dialogue_record

            for turn in range(self.max_turns):
                # 检查学生是否得出正确答案
                self.student_answer = extract_answer(student_response)
                if self._has_correct_answer(self.student_answer, self.correct_answer):
                    if dialogue_record.debug_mode:
                        logger.info("🎉 学生得出正确答案!")
                    dialogue_record.correct = True
                    dialogue_record.final_student_answer = self.student_answer

                    # 完成解题路径
                    if self.use_solution_tree and self.solution_tree:
                        self.complete_student_solution(success=True, final_answer=student_response)
                        if dialogue_record.debug_mode:
                            logger.info("✅ 学生答对了，解题路径已完成")

                    return self.student_answer, self.correct_answer, dialogue_record

                current_turn = turn + 2  # 从第二轮开始

                if dialogue_record.debug_mode:
                    logger.info(f"\n🔄 第 {current_turn} 轮对话:")
                    logger.info("-" * 30)

                # 教师回应
                teacher_input = self._format_teacher_input(problem, self.dialogue_history)
                if dialogue_record.debug_mode:
                    logger.info(f"👨‍🏫 teacher_input [轮次{current_turn}]: {teacher_input}")
                teacher_response_obj = teacher_agent._invoke_agent(teacher_input)
                teacher_response = self._extract_response_text(teacher_response_obj)
                self.dialogue_history.append(("teacher", teacher_response))

                # 检查答案泄露
                leakage_detected = dialogue_record.check_answer_leakage(teacher_response)

                # 检查是否应该结束对话
                if self._should_end_dialogue(teacher_response):
                    if dialogue_record.debug_mode:
                        logger.info("✅ 教师认为解题完成")
                    dialogue_record.final_student_answer = self.student_answer

                    # 完成解题路径（可能成功或失败）
                    if self.use_solution_tree and self.solution_tree:
                        success = self._has_correct_answer(self.student_answer, self.correct_answer)
                        self.complete_student_solution(success=success, final_answer=student_response)
                        if dialogue_record.debug_mode:
                            status = "成功" if success else "失败"
                            logger.info(f"📝 教师结束对话，解题路径{status}")

                    return self.student_answer, self.correct_answer, dialogue_record

                if dialogue_record.debug_mode:
                    logger.info(f"👨‍🏫 教师 [轮次{current_turn}]: {teacher_response}")
                if leakage_detected:
                    if dialogue_record.debug_mode:
                        logger.info("⚠️  检测到答案泄露!")

                # 学生回应
                student_input = self._format_student_input(raw_problem, self.dialogue_history)
                if dialogue_record.debug_mode:
                    logger.info(f"👨‍🎓 student_input [轮次{current_turn}]: {student_input}")
                student_response_obj = self._invoke_agent(student_input)
                student_response = self._extract_response_text(student_response_obj)
                self.dialogue_history.append(("student", student_response))

                # 记录学生解题步骤
                if self.use_solution_tree and self.solution_tree:
                    method_used = self._detect_method_from_response(student_response)
                    self.record_student_step(student_response, method_used=method_used)
                    if dialogue_record.debug_mode:
                        logger.info(f"📝 已记录学生第{current_turn}步解题步骤，方法: {method_used}")

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

            # 完成解题路径（达到最大轮次）
            if self.use_solution_tree and self.solution_tree:
                success = self._has_correct_answer(self.student_answer, self.correct_answer)
                self.complete_student_solution(success=success, final_answer=student_response)
                if dialogue_record.debug_mode:
                    status = "成功" if success else "失败"
                    logger.info(f"📝 达到最大轮次，解题路径{status}")

            return self._get_final_answer(), self.correct_answer, dialogue_record

        except Exception as e:
            if dialogue_record.debug_mode:
                logger.error(f"❌ 对话过程中出现错误: {e}")
            # 在出错时也要完成解题路径
            if self.use_solution_tree and self.solution_tree:
                self.complete_student_solution(success=False, final_answer="错误")
                if dialogue_record.debug_mode:
                    logger.info("📝 对话出错，解题路径标记为失败")

            # 返回一个基本的错误响应
            dialogue_record.final_student_answer = "抱歉，解题过程中出现了技术问题。"
            return "抱歉，解题过程中出现了技术问题。", self.correct_answer, dialogue_record

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

    def _detect_method_from_response(self, response: str) -> str:
        """从学生响应中检测使用的方法"""
        if not response:
            return "unknown"

        response_lower = response.lower()

        # 检测方法类型
        if any(word in response_lower for word in ["equation", "solve for", "variable", "x =", "let x", "algebra"]):
            return "algebraic"
        elif any(word in response_lower for word in
                 ["diagram", "graph", "shape", "angle", "area", "triangle", "circle"]):
            return "geometric"
        elif any(word in response_lower for word in
                 ["calculate", "compute", "number", "digit", "sum", "total", "multiply"]):
            return "computational"
        elif any(word in response_lower for word in ["logic", "reason", "therefore", "because", "since", "if then"]):
            return "logical"
        elif any(word in response_lower for word in ["guess", "try", "maybe", "perhaps", "i think"]):
            return "trial_and_error"
        else:
            return "general"

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
        if self.memory and self.memory.enabled:
            self.memory.add_message("user", input_text)

        if hasattr(self, 'local_agent') and self.local_agent:
            response = self._invoke_local_with_memory(input_text)
        else:
            response = self._invoke_api_with_memory(input_text)

        # 记录助手响应到内存
        if self.memory and self.memory.enabled and response:
            self.memory.add_message("assistant", response)

        return response

    def _invoke_local_with_memory(self, input_text):
        """本地模型调用（支持内存和解题树）"""
        # 构建包含内存上下文和解题树上下文的输入
        memory_context = ""
        if self.memory and self.memory.enabled:
            memory_context = self.memory.get_context()

        # 使用增强的完整提示词构建方法
        full_input = self._build_full_prompt(input_text, memory_context)

        response = self.local_agent.invoke({
            "messages": [{"role": "user", "content": full_input}]
        })
        return response['structured_response'].main_response

    def _invoke_api_with_memory(self, input_text):
        """API模型调用（支持内存和解题树）"""
        memory_context = ""
        if self.memory and self.memory.enabled:
            memory_context = self.memory.get_context()

        # 使用增强的完整提示词构建方法
        full_input = self._build_full_prompt(input_text, memory_context)

        try:
            # 调用agent
            response = self.agent.invoke({
                "messages": [{"role": "user", "content": full_input}]
            }, config=self.agent_config, context=self.context)

            # 处理工具调用情况
            if response and 'messages' in response:
                # 检查最后一条消息是否包含工具调用
                last_message = response['messages'][-1]

                # 如果有工具调用，我们需要处理它们
                if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                    if self.debug_mode:
                        logger.info(f"🛠️ 检测到工具调用: {[tc['name'] for tc in last_message.tool_calls]}")

                    # 对于多智能体对话，我们暂时简化处理：不执行实际工具调用
                    # 而是返回一个提示信息
                    tool_response = "在当前对话模式下，工具调用功能暂时不可用。请直接提供指导或回答。"

                    # 创建工具响应消息
                    tool_messages = []
                    for tool_call in last_message.tool_calls:
                        tool_messages.append({
                            "role": "tool",
                            "content": tool_response,
                            "tool_call_id": tool_call['id']
                        })

                    # 如果需要继续处理工具调用，可以在这里添加更多逻辑
                    # 但目前我们返回一个简化响应
                    return tool_response

                # 正常返回结构化响应
                return response['structured_response'].main_response

            return ""

        except Exception as e:
            if self.debug_mode:
                logger.info(f"❌ API调用错误: {e}")

            # 如果出现工具调用相关错误，回退到简单响应
            if "tool_calls" in str(e) or "tool_call_id" in str(e):
                return "我理解您需要帮助，但在当前设置下，请直接提出您的问题或困惑。"

            raise e

    def _build_input_with_memory(self, input_text, memory_context):
        """构建包含内存上下文的输入"""
        if memory_context:
            return f"{memory_context}\n\nCurrent Question: {input_text}\n\nPlease respond based on the conversation context:"
        else:
            return input_text

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

    def get_memory_stats(self):
        """获取内存统计信息"""
        if self.memory:
            return self.memory.get_stats()
        return {"enabled": False}

    def clear_memory(self):
        """清空内存"""
        if self.memory:
            self.memory.clear()

    def _format_student_input(self, problem, history):
        """格式化学生输入"""
        context = f"Original Problem: {problem}\n\nDialogue History:\n"
        for role, content in history:
            context += f"{role}: {content}\n\n"
        context += 'System: '
        # 根据并行思考开关调整提示
        if self.parallel_thinking_enabled:
            context += STUDENT_STANDARD_PARALLEL_THINKING_PROMPT.format(question=problem)

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


class ExpertStudentAgent(SimpleAgent):
    """学霸Agent"""

    def __init__(self, debug_mode=False):
        super().__init__(agent_type="expert_student", debug_mode=debug_mode)

    def generate_solution_tree(self, problem):
        """生成解题树"""
        prompt = TREE_GENERATE_PROMPT.format(problem)

        response = self._invoke_agent(prompt)
        return self._parse_solution_tree(response, problem)

    def _parse_solution_tree(self, response, problem):
        """解析解题树响应"""
        solution_tree = SolutionTree(problem)

        try:
            # 简单的解析逻辑 - 在实际应用中可以使用更复杂的解析
            if "<SolutionTree>" in response:
                # 提取解决方案路径
                paths_section = response.split("<SolutionPaths>")[1].split("</SolutionPaths>")[0]
                path_blocks = paths_section.split("</Path>")

                for block in path_blocks:
                    if "<Path" in block:
                        # 提取路径信息
                        method = self._extract_site_tag(block, "method")
                        complexity = self._extract_site_tag(block, "complexity")
                        innovation = self._extract_site_tag(block, "innovation")

                        # 提取步骤
                        steps = []
                        intermediate_answers = []
                        step_parts = block.split("<Step")
                        for step_part in step_parts[1:]:
                            if ">" in step_part and "</Step>" in step_part:
                                step_content = step_part.split(">", 1)[1].split("</Step>")[0]
                                steps.append(step_content)
                            if "<IntermediateAnswer>" in step_part and "</IntermediateAnswer>" in step_part:
                                intermediate_content = \
                                step_part.split("<IntermediateAnswer>", 1)[1].split("</IntermediateAnswer>")[0]
                                intermediate_answers.append(intermediate_content)

                        # 提取最终答案
                        final_answer = self._extract_xml_tag(block, "FinalAnswer")

                        solution_tree.add_expert_path({
                            "method": method,
                            "complexity": complexity,
                            "innovation": innovation,
                            "steps": steps,
                            "intermediate_answers": intermediate_answers,
                            "final_answer": final_answer
                        })

        except Exception as e:
            print(f"❌ Error parsing solution tree: {e}")
            # 如果解析失败，创建一个默认的解决方案路径
            solution_tree.add_expert_path({
                "method": "algebraic",
                "complexity": "medium",
                "innovation": "medium",
                "steps": ["Apply standard algebraic approach", "Solve step by step"],
                "intermediate_answers": [],
                "final_answer": "[[Answer will be determined]]"
            })

        return solution_tree

    def _extract_xml_tag(self, text, tag_name):
        """提取XML标签内容"""
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"

        if start_tag in text and end_tag in text:
            return text.split(start_tag)[1].split(end_tag)[0].strip()
        return ""

    def _extract_site_tag(self, text, tag_name):
        """提取XML标签内容"""
        start_tag = f'{tag_name}="'
        end_tag = f'"'

        if start_tag in text and end_tag in text:
            return text.split(start_tag)[1].split(end_tag)[0].strip()
        return ""


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



