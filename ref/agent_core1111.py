from typing import Dict, Any, Optional
from types import SimpleNamespace

from v8.utils.evaluator import logger, extract_answer
from v8.function.api_client import create_api_client, RawAPIClient, RawOllamaClient
from v8.utils.MARIO_EVAL.demo import is_equiv_MATH
from v8.function.memory import SmartSummaryMemory, SummaryConfig
from v8.function.solution_tree import SolutionTree
from v8.function.cognitive import StudentCognitiveState
from v8.function.middleware import CustomMiddleware, ModelConfig, MiddlewareFunc
from v8.function.contex_fun import Context
from v8.function.format_fun import ResponseFormat
from v8.function.record_eval import DialogueRecord
from v8.prompt.system import *
from v8.prompt.dialogue_tree_parallel import *
from v8.prompt.dialogue_socratic import *


class SimpleLocalAgent:
    """简化版本地Agent，支持核心功能"""

    def __init__(self, agent_type="student", debug_mode=False):
        self.agent_type = agent_type
        self.model = None
        self.system_prompt = ""
        self.context = {}
        self.dialogue_history = []
        self.debug_mode = debug_mode
        self.last_full_prompt = ""  # 记录最后一次完整prompt

    def init_model(self, model_config):
        """初始化本地模型"""
        self.model = create_api_client(model_config)

    def set_system_prompt(self, prompt):
        """设置系统提示词"""
        self.system_prompt = prompt

    def set_context(self, **kwargs):
        """设置上下文"""
        self.context.update(kwargs)

    def invoke(self, input_dict, config=None, context=None):
        """调用Agent"""
        user_input = self._extract_input(input_dict)
        full_prompt = user_input

        # 构建完整提示词
        # full_prompt = self._build_full_prompt(full_prompt)
        self.last_full_prompt = full_prompt  # 保存完整prompt

        # Debug输出
        if self.debug_mode:
            logger.info("\n" + "=" * 80)
            logger.info(f"🔍 DEBUG - {self.agent_type.upper()} AGENT FULL PROMPT:")
            logger.info("=" * 80)
            logger.info(full_prompt)
            logger.info("=" * 80 + "\n")

        # 调用模型 - 使用消息格式
        messages = [{"role": "user", "content": full_prompt}]
        if self.system_prompt:
            messages.insert(0, {"role": "system", "content": self.system_prompt})
        
        response = self.model.invoke(messages)

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
            prompt_parts.append(f"user: {self.system_prompt}")  # dddd

        # 上下文信息
        if self.context:
            context_str = ", ".join([f"{k}: {v}" for k, v in self.context.items()])
            prompt_parts.append(f"Context: {context_str}")

        # 对话历史
        if self.dialogue_history:
            d_list = []
            for role, content in self.dialogue_history[-6:]:  # 最近3轮对话   # dddd
                if role == self.agent_type:
                    d_list.append(f"assistant\n{content}")
                else:
                    d_list.append(f"user\n{content}")
            history_str = "\n".join(d_list)
            prompt_parts.append(f"\n{history_str}")

        # 当前输入
        prompt_parts.append(f"\nuser\n{user_input}")
        prompt_parts.append("<im_end>\n<im_start>assistant")

        return "\n\n".join(prompt_parts)
    

class SimpleAgent(object):
    """增强的Agent类，支持多模式和本地调用"""

    def __init__(self, agent_type: str = "student", debug_mode: bool = False):
        """初始化agent

        Args:
            agent_type: "student" 或 "teacher"
        """
        self.agent_type = agent_type
        self.prompt_sys = None
        self.student_response = None
        self.teacher_response = None

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

    def _build_full_prompt(self, user_input, memory_context, prompt_type='api'):
        """构建完整提示词（增强版）"""
        if self.agent_type == "student":
            if prompt_type == 'local':
                prompt_parts = ['<|im_start|>\n']

                # 系统提示词
                if self.prompt_sys and self.agent_type != 'student':
                    # 处理字符串类型的系统提示词
                    if isinstance(self.prompt_sys, str):
                        # 如果是预定义的提示词变量名，尝试获取其值
                        if self.prompt_sys in globals():
                            system_prompt = globals()[self.prompt_sys]
                        else:
                            system_prompt = self.prompt_sys
                    else:
                        system_prompt = str(self.prompt_sys)
                    prompt_parts.append(f"{system_prompt}\n\n")

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
                if self.dialogue_history:
                    if self.dialogue_history[-1][1] == user_input:
                        dialogue_history = self.dialogue_history[:-1]
                    else:
                        dialogue_history = self.dialogue_history
                    d_list = []
                    for role, content in dialogue_history[-6:]:  # 最近3轮对话   # dddd
                        if role == self.agent_type:
                            d_list.append(f"assistant\n{content}\n\n")
                        else:
                            d_list.append(f"user\n{content}\n\n")
                    history_str = "\n".join(d_list)
                    prompt_parts.append(f"\n{history_str}")

                # 当前输入
                if user_input and len(user_input):
                    prompt_parts.append(f"\nuser\n{user_input}\n")
                prompt_parts.append("<|im_end|>\n<|im_start|>assistant")
                return "\n\n".join(prompt_parts)
            elif prompt_type == 'api':
                # Build messages list in OpenAI format: [{"role": "system/user/assistant", "content": "..."}, ...]
                messages = [{"role": "system", "content": self.prompt_sys}]
                system_content = None
                
                # 1. Dialogue history (previous user and assistant messages)
                if self.dialogue_history:
                    # 如果最后一条历史记录是当前输入，则排除它
                    if self.dialogue_history and self.dialogue_history[-1][1] == user_input:
                        dialogue_history = self.dialogue_history[:-1]
                    else:
                        dialogue_history = self.dialogue_history
                    
                    # 转换对话历史为消息格式（保留最近6轮）
                    for role, content in dialogue_history[-6:]:
                        # 转换角色：student -> assistant, teacher -> user (for student agent)
                        if role == self.agent_type:
                            messages.append({"role": "assistant", "content": content})
                        else:
                            messages.append({"role": "user", "content": content})
                
                # 2. Current user input
                if user_input and len(user_input):
                    messages.append({"role": "user", "content": user_input})

                # 3. Other prompts
                if self.use_cognitive_state or self.use_solution_tree or len(memory_context):
                    # 认知状态（如果是学生且启用）
                    if (self.agent_type == "student" and self.use_cognitive_state and
                            self.cognitive_state and hasattr(self.cognitive_state, 'get_prompt_context')):
                        cognitive_context = self.cognitive_state.get_prompt_context()
                        if messages[0]['role'] == 'system':
                            messages[0]['content'] += f"\n\nStudent Profile: {cognitive_context}"

                    # 解题树上下文（如果启用）
                    if self.use_solution_tree and self.solution_tree:
                        try:
                            tree_context = self._build_solution_tree_context()
                            if tree_context:
                                if messages[-1]['role'] == 'assistant':
                                    messages[-1]['content'] += f"\n\nSolution Context: {tree_context}"
                        except Exception as e:
                            if self.debug_mode:
                                logger.info(f"❌ 解题树上下文构建失败: {e}")
                
                return messages
        elif self.agent_type == "teacher":
            return user_input
        else:
            raise NotImplementedError

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
            # API模型配置 - 使用原始API客户端
            self.model = create_api_client(model_config)
            print(f"✅ 初始化 {model_name} API模型: {model_config.model_name}")
            # 不再需要中间件，因为使用原始API调用
            self.middleware_er = None

    def _add_debug_middleware(self):
        """为API模型添加debug中间件 - 已移除，使用原始API调用"""
        pass

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

    def agent_init(self, model_config, prompt_sys_name=None, teacher_response=None, student_response=None,
                   context_schema=Context, response_format=ResponseFormat, **kwargs):
        """初始化agent配置"""
        self.context_schema = context_schema
        self.response_format = response_format

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
        self.teacher_response = teacher_response
        self.student_response = student_response

        # 根据功能开关调整系统提示词
        self._adjust_prompt_based_on_settings()

        # 中间件不再需要，因为使用原始API调用
        # middleware参数保留用于兼容性，但不使用

        if model_config.model_type == "local":
            # 本地模型初始化
            self.local_agent.set_system_prompt(self.prompt_sys)
            if self.context:
                self.local_agent.set_context(**self.context.__dict__)
        else:
            # 对于API模型，使用简化的agent创建方式（原始API调用）
            self.agent = self._create_simple_agent()
            if self.debug_mode:
                print("✅ 使用简化Agent（原始API调用）")

    def _create_simple_agent(self):
        """创建不包含工具的简单对话agent - 使用原始API调用"""
        # 包装成类似agent的接口
        class SimpleAgentWrapper:
            def __init__(self, model, prompt_sys, debug_mode=False):
                self.model = model
                self.prompt_sys = prompt_sys
                self.debug_mode = debug_mode

            def invoke(self, input_dict, config=None, context=None):
                try:
                    api_messages = []
                    # 提取用户输入
                    messages = input_dict.get("messages", [])
                    for msg in messages:
                        if isinstance(msg, dict) and 'content' in msg:
                            api_messages.append({"role": msg['role'], "content": msg['content']})
                    
                    # 调用模型
                    response = self.model.invoke(api_messages)
                    
                    # 返回与现有代码兼容的格式
                    return response

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

        return SimpleAgentWrapper(self.model, self.prompt_sys, self.debug_mode)

    def get_prompt_str(self, prompt_name):
        if prompt_name:
            if isinstance(prompt_name, str):
                # 如果是预定义的提示词变量名，尝试获取其值
                if prompt_name in globals():
                    base_prompt = globals()[prompt_name]
                else:
                    base_prompt = prompt_name
            else:
                base_prompt = str(prompt_name)
            return base_prompt
        else:
            return None

    def _adjust_prompt_based_on_settings(self):
        """根据功能开关调整系统提示词"""
        base_prompt = self.get_prompt_str(self.prompt_sys)
        self.student_response = self.get_prompt_str(self.student_response)
        self.teacher_response = self.get_prompt_str(self.teacher_response)

        if self.agent_type == "student":

            if self.parallel_thinking_enabled:
                base_prompt += "\n\n" + PARALLEL_THINKING_PROMPT

            self.prompt_sys = base_prompt

        elif self.agent_type == "teacher":

            if self.correct_answer:
                base_prompt += f"\n\nYou know the correct solution is: {self.correct_answer}. But DO NOT reveal this answer directly to the student. Guide the student to discover it themselves."

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

    def check_answer(self, student_response):
        self.student_answer = extract_answer(student_response)
        correctness = self._has_correct_answer(self.student_answer, self.correct_answer)
        if not correctness:
            self.student_answer = extract_answer(self.student_answer)
            correctness = self._has_correct_answer(self.student_answer, self.correct_answer)
        return correctness

    def multi_agent_chat_explicit(self, teacher_agent, problem: str, raw_problem: str, correct_answer: str,
                                  dialogue_record: DialogueRecord, **kwargs):
        """模式1: 显式交互 - 教师和学生直接对话"""
        if dialogue_record.debug_mode:
            logger.info(f"\n🎯 开始解题: {problem}")
            logger.info("=" * 50)

        # 设置教师知道的正确答案
        teacher_agent.set_correct_answer(correct_answer)
        teacher_agent._adjust_prompt_based_on_settings()  # 重新调整提示词

        # 重置对话历史
        self.dialogue_history = [('teacher', problem)]
        self.current_turn = 0

        # 初始化内存管理
        global_memory = None
        teacher_memory = None

        # 检查是否启用摘要功能
        if hasattr(self, 'memory') and self.memory and getattr(self.memory, 'enabled', False):
            global_memory = self.memory
            global_memory.debug_mode = dialogue_record.debug_mode
            global_memory.add_message('teacher', problem)
            if dialogue_record.debug_mode:
                logger.info("🗂 已将初始问题加入学生摘要内存")

        if hasattr(teacher_agent, 'memory') and teacher_agent.memory and getattr(teacher_agent.memory, 'enabled', False):
            teacher_memory = teacher_agent.memory
            teacher_memory.debug_mode = dialogue_record.debug_mode
            teacher_memory.add_message('teacher', problem)
            if dialogue_record.debug_mode:
                logger.info("🗂 已将初始问题加入教师摘要内存")

        try:
            # 学生首次尝试
            student_prompt_type = 'local' if (hasattr(self, 'local_agent') and self.local_agent) else 'api'

            # 在构建输入前检查是否需要生成摘要
            self._check_and_generate_summary(global_memory, teacher_memory, dialogue_record.debug_mode)

            student_input = self._format_student_input(raw_problem, self.dialogue_history, prompt_type=student_prompt_type)
            student_response_obj = self._invoke_agent(student_input)
            student_response = self._extract_response_text(student_response_obj)
            self.dialogue_history.append(("student", student_response))

            # 将学生回复加入摘要内存
            self._add_to_memory(global_memory, teacher_memory, 'student', student_response, dialogue_record.debug_mode)

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
                dialogue_record.first_correct = self.check_answer(student_response)

                for turn in range(self.max_turns):
                    # 检查学生是否得出正确答案
                    if (turn == 0 and dialogue_record.first_correct) or (self.check_answer(student_response)):
                        if dialogue_record.debug_mode:
                            logger.info("🎉 学生得出正确答案!")
                        dialogue_record.correct = True
                        dialogue_record.final_student_answer = self.student_answer

                        # 更新认知状态（最后一轮，成功解决）
                        if self.use_cognitive_state and self.cognitive_state:
                            try:
                                errors = dialogue_record._extract_errors()
                                method_used = self._detect_method_from_response(student_response)
                                self.cognitive_state.update_based_on_interaction(
                                    problem=problem,
                                    student_approach=student_response,
                                    errors=errors,
                                    method_used=method_used,
                                    success=True  # 成功解决问题
                                )
                            except Exception as e:
                                if dialogue_record.debug_mode:
                                    logger.error(f"更新最终认知状态时出错: {e}")

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
                    # teacher_input = student_response
                    if self.parallel_thinking_enabled and turn == 0:
                        teacher_response = STUDENT_STANDARD_PARALLEL_THINKING_PROMPT.format(question=raw_problem)
                    else:
                        # 确定教师agent使用的模型类型
                        teacher_prompt_type = 'local' if (hasattr(teacher_agent, 'local_agent') and teacher_agent.local_agent) else 'api'
                        teacher_input = teacher_agent._format_teacher_input(self.dialogue_history, prompt_type=teacher_prompt_type)
                        teacher_response_obj = teacher_agent._invoke_agent(teacher_input)
                        teacher_response = self._extract_response_text(teacher_response_obj)

                    self.dialogue_history.append(("teacher", teacher_response))

                    # 检查答案泄露
                    leakage_detected = dialogue_record.check_answer_leakage(teacher_response, self.correct_answer)

                    if dialogue_record.debug_mode:
                        logger.info(f"👨‍🏫 教师 [轮次{current_turn}]: {teacher_response}")
                    if leakage_detected:
                        if dialogue_record.debug_mode:
                            logger.info("⚠️  检测到答案泄露!")

                # 在学生回应前检查是否需要生成摘要
                self._check_and_generate_summary(global_memory, teacher_memory, dialogue_record.debug_mode)

                # 学生回应
                student_prompt_type = 'local' if (hasattr(self, 'local_agent') and self.local_agent) else 'api'
                student_input = self._format_student_input(raw_problem, self.dialogue_history,
                                                           prompt_type=student_prompt_type)
                student_response_obj = self._invoke_agent(student_input)
                student_response = self._extract_response_text(student_response_obj)
                self.dialogue_history.append(("student", student_response))
                self._add_to_memory(global_memory, teacher_memory, 'student', student_response, dialogue_record.debug_mode)

                # 记录学生解题步骤
                if self.use_solution_tree and self.solution_tree:
                    method_used = self._detect_method_from_response(student_response)
                    self.record_student_step(student_response, method_used=method_used)
                    if dialogue_record.debug_mode:
                        logger.info(f"📝 已记录学生第{current_turn}步解题步骤，方法: {method_used}")

                # 分析学生回复
                parallel_count, path_count = dialogue_record.analyze_student_response(student_response)

                # 记录本轮对话
                turn_data = {
                    "answer": self.correct_answer,
                    "answer_response": correct_answer,
                    "turn": current_turn,
                    "student_response": student_response,
                    "teacher_response": teacher_response,
                    "teacher_intent": self._analyze_teacher_intent(teacher_response),
                    "parallel_thinking_count": parallel_count,
                    "thinking_paths_count": path_count,
                    "answer_leakage": leakage_detected
                }
                dialogue_record.add_turn(turn_data)

                # 在每轮对话后更新认知状态（当前轮不是最后一轮）
                if self.use_cognitive_state and self.cognitive_state:
                    # 从run.py导入辅助函数
                    errors = dialogue_record._extract_errors()
                    method_used = self._detect_method_from_response(student_response)
                    try:
                        self.cognitive_state.update_based_on_interaction(
                            problem=problem,
                            student_approach=student_response,
                            errors=errors,
                            method_used=method_used,
                            success=False  # 还在对话中，未最终解决
                        )
                    except Exception as e:
                        if dialogue_record.debug_mode:
                            logger.error(f"更新认知状态时出错: {e}")

                # 在每轮对话后更新认知状态（当前轮不是最后一轮）
                if self.cognitive_state:
                    errors = dialogue_record._extract_errors()
                    method_used = self._detect_method_from_response(student_response)
                    self.cognitive_state.update_based_on_interaction(
                        problem=problem,
                        student_approach=student_response,
                        errors=errors,
                        method_used=method_used,
                        success=False  # 还在对话中，未最终解决
                    )
                if dialogue_record.debug_mode:
                    logger.info(f"👨‍🎓 学生 [轮次{current_turn}]: {student_response}")

                self.current_turn = current_turn

            # 设置最终答案
            dialogue_record.final_student_answer = self.student_answer
            success = self._has_correct_answer(self.student_answer, self.correct_answer)

            # 更新最终认知状态（达到最大轮次）
            if self.use_cognitive_state and self.cognitive_state:
                try:
                    errors = dialogue_record._extract_errors()
                    method_used = self._detect_method_from_response(student_response)
                    self.cognitive_state.update_based_on_interaction(
                        problem=problem,
                        student_approach=student_response,
                        errors=errors,
                        method_used=method_used,
                        success=success  # 基于最终答案是否正确
                    )
                except Exception as e:
                    if dialogue_record.debug_mode:
                        logger.error(f"更新最终认知状态时出错: {e}")

            # 完成解题路径（达到最大轮次）
            if self.use_solution_tree and self.solution_tree:
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


    def _check_and_generate_summary(self, global_memory, teacher_memory, debug_mode):
        """检查并生成对话摘要"""
        try:
            # 检查学生内存是否需要生成摘要
            if global_memory and getattr(global_memory, 'enabled', False):
                if global_memory._should_generate_summary():
                    global_memory._generate_summary()
                    if debug_mode:
                        logger.info("📝 学生内存已生成新摘要")

            # 检查教师内存是否需要生成摘要
            if teacher_memory and getattr(teacher_memory, 'enabled', False):
                if teacher_memory._should_generate_summary():
                    teacher_memory._generate_summary()
                    if debug_mode:
                        logger.info("📝 教师内存已生成新摘要")

        except Exception as e:
            if debug_mode:
                logger.warning(f"生成摘要时出错: {e}")


    def _add_to_memory(self, global_memory, teacher_memory, role, content, debug_mode):
        """将消息添加到内存"""
        try:
            if global_memory and getattr(global_memory, 'enabled', False):
                global_memory.add_message(role, content)

            if teacher_memory and getattr(teacher_memory, 'enabled', False):
                teacher_memory.add_message(role, content)

        except Exception as e:
            if debug_mode:
                logger.warning(f"添加消息到内存时出错: {e}")

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

        # 更新系统提示词以包含工具说明
        updated_prompt = self.prompt_sys + "\n\nYou can use the ask_teacher tool when you need guidance."

        # 学生自主解题，可在需要时调用教师工具
        # 注意：工具调用模式在原始API中需要特殊处理，这里简化为直接调用
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
        # 简单的工具函数，不使用LangChain装饰器
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
        """调用agent并返回响应
        
        Args:
            input_text: 可以是字符串或消息列表（OpenAI格式）
                       - 字符串：用于学生agent或本地模型
                       - 消息列表：用于教师agent的API调用（已格式化）
        """
        # 检查输入是否已经是消息列表格式（API调用）
        if isinstance(input_text, list) and len(input_text) > 0 and isinstance(input_text[0], dict):
            # 输入已经是消息列表，直接调用API
            if hasattr(self, 'local_agent') and self.local_agent:
                # 本地模型不应该收到消息列表，转换为字符串
                # 提取最后一条用户消息
                user_content = ""
                for msg in reversed(input_text):
                    if msg.get("role") == "user":
                        user_content = msg.get("content", "")
                        break
                input_text = user_content
                response = self._invoke_local_with_memory(input_text)
            else:
                # API调用，直接使用消息列表
                try:
                    response = self.agent.invoke({
                        "messages": input_text
                    }, config=self.agent_config, context=self.context)
                    
                    if response and 'messages' in response:
                        response_text = response['structured_response'].main_response
                    else:
                        response_text = ""
                    
                    # 记录到内存
                    if self.memory and self.memory.enabled:
                        # 提取用户输入（最后一条用户消息）
                        for msg in reversed(input_text):
                            if msg.get("role") == "user":
                                self.memory.add_message("user", msg.get("content", ""))
                                break
                        self.memory.add_message("assistant", response_text)
                    
                    return response_text
                except Exception as e:
                    if self.debug_mode:
                        logger.info(f"❌ API调用错误: {e}")
                    raise e
        else:
            # 输入是字符串，使用标准流程
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
        full_input = self._build_full_prompt(input_text, memory_context, 'local')

        response = self.local_agent.invoke({
            "messages": [{"role": "user", "content": full_input}]
        })
        return response['structured_response'].main_response

    def _invoke_api_with_memory(self, input_text):
        """API模型调用（支持内存和解题树）"""
        memory_context = ""
        if self.memory and self.memory.enabled:
            memory_context = self.memory.get_context()

        # 使用增强的完整提示词构建方法 - 返回消息列表（OpenAI格式）
        messages = self._build_full_prompt(input_text, memory_context, 'api')

        try:
            # 调用agent - messages已经是正确格式
            response = self.agent.invoke({
                "messages": messages
            }, config=self.agent_config, context=self.context)

            if response and 'messages' in response:
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

    def _format_teacher_input(self, history, prompt_type='api'):
        """格式化教师输入，支持本地和API调用
        
        Args:
            history: 对话历史列表，格式为 [(role, content), ...]
            prompt_type: 'api' 或 'local'
        
        Returns:
            - 对于 'api': 返回消息列表 [{"role": "system/user/assistant", "content": "..."}, ...]
            - 对于 'local': 返回字符串格式的提示词
        """
        if prompt_type == 'api':
            # Build messages list in OpenAI format: [{"role": "system/user/assistant", "content": "..."}, ...]
            messages = []
            system_prompt = None
            # 1. System prompt (if exists)
            if self.prompt_sys:
                # 处理字符串类型的系统提示词
                if isinstance(self.prompt_sys, str):
                    # 如果是预定义的提示词变量名，尝试获取其值
                    if self.prompt_sys in globals():
                        system_prompt = globals()[self.prompt_sys]
                    else:
                        system_prompt = self.prompt_sys
                else:
                    system_prompt = str(self.prompt_sys + "\n\n")
            
            # 2. Dialogue history (previous user and assistant messages)
            if history:
                # 转换对话历史为消息格式（保留最近6轮）
                for role, content in history[-6:]:
                    # 转换角色：teacher -> assistant, student -> user (for teacher agent)
                    if role == self.agent_type:
                        messages.append({"role": "assistant", "content": content})
                    else:
                        messages.append({"role": "user", "content": content})
                
                if messages[-1]['role'] == 'user':
                    if self.socratic_teaching_enabled:
                        messages[-1]['content'] += f"\n\n{TEACHER_RESPONSE_TEACHING_SOCRATIC}"
                    else:
                        messages[-1]['content'] += f"\n\n{self.teacher_response}"
                
                if messages[0]['role'] == 'user':
                    if system_prompt:
                        messages[0]['content'] = system_prompt + messages[0]['content']
                elif messages[0]['role'] == 'assistant':
                    if system_prompt:
                        messages.insert(0, {"role": "user", "content": system_prompt})
            
            return messages
        
        elif prompt_type == 'local':
            # 本地模型格式：返回字符串
            context = ''
            d_list = []
            for role, content in history[-6:]:  # 最近3轮对话
                if role == self.agent_type:
                    d_list.append(f"assistant\n{content}\n\n")
                else:
                    d_list.append(f"user\n{content}\n\n")
            history_str = "\n".join(d_list)
            context += f"\n{history_str}"
            
            # 根据苏格拉底教学开关调整提示
            if self.socratic_teaching_enabled:
                context += TEACHER_RESPONSE_TEACHING_SOCRATIC
            else:
                context += TEACHER_RESPONSE_TEACHING

            return context
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}. Must be 'api' or 'local'.")

    def _format_student_input(self, problem, history, prompt_type='api'):
        """格式化教师输入，支持本地和API调用

        Args:
            history: 对话历史列表，格式为 [(role, content), ...]
            prompt_type: 'api' 或 'local'

        Returns:
            - 对于 'api': 返回消息列表 [{"role": "system/user/assistant", "content": "..."}, ...]
            - 对于 'local': 返回字符串格式的提示词
        """
        if prompt_type == 'api':
            # Build messages list in OpenAI format: [{"role": "system/user/assistant", "content": "..."}, ...]
            messages = []
            system_prompt = None
            # 1. System prompt (if exists)
            if self.prompt_sys:
                # 处理字符串类型的系统提示词
                if isinstance(self.prompt_sys, str):
                    # 如果是预定义的提示词变量名，尝试获取其值
                    if self.prompt_sys in globals():
                        system_prompt = globals()[self.prompt_sys]
                    else:
                        system_prompt = self.prompt_sys
                else:
                    system_prompt = str(self.prompt_sys + "\n\n")

            # 2. Dialogue history (previous user and assistant messages)
            if history:
                # 转换对话历史为消息格式（保留最近6轮）
                for role, content in history[-6:]:
                    # 转换角色：teacher -> assistant, student -> user (for teacher agent)
                    if role == self.agent_type:
                        messages.append({"role": "assistant", "content": content})
                    else:
                        messages.append({"role": "user", "content": content})

                if messages[-1]['role'] == 'user':
                    messages[-1]['content'] += "\n\n" + self.student_response.format(question=problem)
                if self.parallel_thinking_enabled:
                    messages[-1]['content'] += STUDENT_STANDARD_PARALLEL_THINKING_PROMPT.format(question=problem)

                if messages[0]['role'] == 'user':
                    if system_prompt:
                        messages[0]['content'] = system_prompt + messages[0]['content']
                elif messages[0]['role'] == 'assistant':
                    if system_prompt:
                        messages.insert(0, {"role": "user", "content": system_prompt})

            return messages

        elif prompt_type == 'local':
            # 本地模型格式：返回字符串
            context = ''
            d_list = []
            for role, content in history[-6:]:  # 最近3轮对话
                if role == self.agent_type:
                    d_list.append(f"assistant\n{content}\n\n")
                else:
                    d_list.append(f"user\n{content}\n\n")
            history_str = "\n".join(d_list)
            context += f"\n{history_str}"

            # 根据苏格拉底教学开关调整提示
            if self.socratic_teaching_enabled:
                context += TEACHER_RESPONSE_TEACHING_SOCRATIC
            else:
                context += TEACHER_RESPONSE_TEACHING

            return context
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}. Must be 'api' or 'local'.")

    def get_memory_stats(self):
        """获取内存统计信息"""
        if self.memory:
            return self.memory.get_stats()
        return {"enabled": False}

    def clear_memory(self):
        """清空内存"""
        if self.memory:
            self.memory.clear()

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


# LocalModelWrapper is now replaced by RawOllamaClient in api_client.py


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
    


