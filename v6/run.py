from function.agent_IO import ModelConfig, ExperimentRecorder, ExpertStudentAgent, SimpleAgent, SummaryConfig, StudentCognitiveState, DialogueRecord
from utils.evaluator import *
from utils.dataset.parallel_thinking_sft_dataset import RawDataset

from langchain.chat_models import init_chat_model
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

device_name = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device_name}")


class MultiAgentSystem:
    """多智能体系统管理器"""

    def __init__(self, config):
        self.config = config
        self.student_agent = None
        self.teacher_agent = None
        self.expert_agent = None
        self.summary_llm = None
        self.current_solution_tree = None
        exp_name = f"math_tutoring_explicit_interaction" if config.agent.explicit_interaction else "math_tutoring_explicit"
        self.experiment_recorder = ExperimentRecorder(exp_name)

    def initialize_agents(self):
        """初始化学生和教师智能体"""
        debug_mode = self.config.agent.get("debug_mode", False)

        # 学生模型配置
        student_model_config = ModelConfig(
            model_type=self.config.model.model_type_student,
            model_name=self.config.model.api_name_student,
            base_url=self.config.model.base_url_student if hasattr(self.config.model, 'base_url_student') else None,
            temperature=self.config.model.temperature_student,
            max_tokens=self.config.model.max_tokens_student
        )

        # 教师模型配置
        teacher_model_config = ModelConfig(
            model_type=self.config.model.model_type_teacher,
            model_name=self.config.model.api_name_teacher,
            base_url=self.config.model.base_url_teacher if hasattr(self.config.model, 'base_url_teacher') else None,
            temperature=self.config.model.temperature_teacher,
            max_tokens=self.config.model.max_tokens_teacher
        )

        # 初始化学霸Agent（如果启用解题树）
        if self.config.agent.get("use_solution_tree", False):
            self.expert_agent = ExpertStudentAgent(
                debug_mode=self.config.agent.get("debug_mode", False)
            )
            expert_model_config = ModelConfig(
                model_type=self.config.model.model_type_expert,  # 使用教师模型配置
                model_name=self.config.model.api_name_expert,
                base_url=self.config.model.base_url_expert if hasattr(self.config.model, 'base_url_student') else None,
                temperature=self.config.model.temperature_expert,
                max_tokens=self.config.model.max_tokens_expert
            )
            self.expert_agent.model_init(expert_model_config, 'expert')
            self.expert_agent.config_create("thread_id", "expert_1")
            self.expert_agent.agent_init(
                expert_model_config,
                prompt_sys_name=self.config.agent.expert_sys_prompt,
                tools_list=[]
            )
            logger.info(f"✅ Expert agent initialized: {self.config.model.api_name_student}")

        # 初始化学生智能体
        self.student_agent = SimpleAgent(agent_type="student", debug_mode=debug_mode)
        self.student_agent.model_init(student_model_config, 'student')
        self.student_agent.config_create("thread_id", "student_1")

        # 初始化教师智能体
        self.teacher_agent = SimpleAgent(agent_type="teacher", debug_mode=debug_mode)
        self.teacher_agent.model_init(teacher_model_config, 'teacher')
        self.teacher_agent.config_create("thread_id", "teacher_1")

        # 配置学生认知状态
        self._setup_cognitive_state()

        # 配置对话摘要功能
        self._setup_conversation_summary()

        # 配置智能体参数
        student_kwargs = {
            "max_turns": self.config.agent.get("max_turns", 5),
            "parallel_thinking": self.config.agent.get("parallel_thinking", False),
        }

        teacher_kwargs = {
            "socratic_teaching": self.config.agent.get("socratic_teaching", False),
        }

        # 工具配置
        student_tools = []
        teacher_tools = []

        # if self.config.agent.get("parallel_thinking", False):
        #     student_tools.append(parallel_thinking)
        #
        # if self.config.agent.get("socratic_teaching", True):
        #     teacher_tools.append(socratic_questioning)
        #     teacher_tools.append(math_concept_explainer)

        if self.config.agent.get("use_solution_tree", False):
            teacher_kwargs["prompt_solution_tree"] = self.config.agent.teacher_tree_sys_prompt
            student_kwargs["prompt_solution_tree"] = self.config.agent.student_tree_sys_prompt

        self.student_agent.agent_init(
            student_model_config,
            prompt_sys_name=self.config.agent.student_sys_prompt,
            tools_list=student_tools,
            **student_kwargs
        )
        self.teacher_agent.agent_init(
            teacher_model_config,
            prompt_sys_name=self.config.agent.teacher_sys_prompt,
            tools_list=teacher_tools,
            **teacher_kwargs
        )

        # 启用摘要功能
        summary_config = SummaryConfig(
            enabled=self.config.agent.get("conversation_summary", False),
            max_turns=self.config.agent.get("summary_max_turns", 8),
            max_token_limit=self.config.agent.get("summary_max_tokens", 1500)
        )

        summary_enabled = self.config.agent.get("conversation_summary", False)
        if summary_enabled:
            self.student_agent.enable_conversation_summary(summary_config, self.summary_llm)
            self.teacher_agent.enable_conversation_summary(summary_config, self.summary_llm)
            logger.info(f"✅ 对话摘要功能已启用 (max_turns: {summary_config.max_turns})")
        else:
            logger.info("✅ 对话摘要功能已禁用")

        # 设置上下文
        self.student_agent.context_set(
            user_id="Jack",
            user_role="student",
            parallel_thinking=student_kwargs["parallel_thinking"],
            conversation_mode="explicit" if self.config.agent.get("explicit_interaction", True) else "tool_based"
        )
        self.teacher_agent.context_set(
            user_id="Professor Smith",
            user_role="teacher",
            socratic_teaching=teacher_kwargs["socratic_teaching"]
        )

        logger.info("✅ Multi-agent system initialized successfully!")
        logger.info(f"   Student: {self.config.model.api_name_student}")
        logger.info(f"   Teacher: {self.config.model.api_name_teacher}")
        logger.info(f"   Expert: {'Enabled' if self.expert_agent else 'Disabled'}")
        logger.info(f"   Cognitive State: {'Enabled' if self.config.agent.get('use_cognitive_state', False) else 'Disabled'}")
        logger.info(f"   Solution Tree: {'Enabled' if self.config.agent.get('use_solution_tree', False) else 'Disabled'}")
        logger.info(
            f"   Mode: {'Explicit Interaction' if self.config.agent.get('explicit_interaction', True) else 'Tool-based'}")
        logger.info(f"   Parallel Thinking: {student_kwargs['parallel_thinking']}")
        logger.info(f"   Socratic Teaching: {teacher_kwargs['socratic_teaching']}")

        return self.student_agent, self.teacher_agent

    def _setup_cognitive_state(self):
        """配置学生认知状态"""
        if self.config.agent.get("use_cognitive_state", False):
            cognitive_state = StudentCognitiveState(
                carelessness_level=self.config.agent.get("carelessness_level", 5),
                math_background=self.config.agent.get("math_background", "intermediate"),
                response_style=self.config.agent.get("response_style", "thoughtful"),
                preferred_method=self.config.agent.get("preferred_method", "balanced"),
                learning_style=self.config.agent.get("learning_style", "reading-writing")
            )
            self.student_agent.set_cognitive_state(cognitive_state)

    def generate_solution_tree(self, problem):
        """生成解题树"""
        if self.expert_agent and hasattr(self.expert_agent, 'generate_solution_tree'):
            self.current_solution_tree = self.expert_agent.generate_solution_tree(problem)

            # 设置解题树到学生和教师agent
            if self.current_solution_tree:
                self.student_agent.set_solution_tree(self.current_solution_tree)
                self.teacher_agent.set_solution_tree(self.current_solution_tree)

            return self.current_solution_tree
        return None

    def update_cognitive_state(self, problem, student_approach, errors, method_used, success):
        """更新学生认知状态"""
        if (self.config.agent.get("use_cognitive_state", False) and
                self.student_agent.cognitive_state):
            self.student_agent.cognitive_state.update_based_on_interaction(
                problem, student_approach, errors, method_used, success
            )

    def get_cognitive_state(self):
        """获取当前认知状态"""
        if (self.config.agent.get("use_cognitive_state", False) and
                self.student_agent.cognitive_state):
            return self.student_agent.cognitive_state.to_dict()
        return None

    def update_cognitive_state_based_on_dialogue(self, problem, dialogue_record, success):
        """基于对话记录更新认知状态"""
        if not self.config.agent.get("use_cognitive_state", False):
            return

        if not self.student_agent or not self.student_agent.cognitive_state:
            return

        # 分析学生的方法和错误
        student_approach = _analyze_student_approach(dialogue_record)
        errors = _extract_errors(dialogue_record)
        method_used = _detect_method_used(dialogue_record)
        response_characteristics = _analyze_student_response_characteristics(dialogue_record)

        # 更新认知状态
        self.student_agent.cognitive_state.update_based_on_interaction(
            problem, student_approach, errors, method_used, success
        )

        # 更新回复风格（基于观察）
        if response_characteristics != "neutral":
            self.student_agent.cognitive_state.response_style = response_characteristics

    def analyze_student_progress(self, dialogue_record):
        """分析学生进步情况"""
        if not self.config.agent.get("use_cognitive_state", False):
            return None

        if not self.student_agent or not self.student_agent.cognitive_state:
            return None

        cognitive_state = self.student_agent.cognitive_state.to_dict()

        progress_info = {
            "carelessness_trend": "stable",
            "background_improvement": "none",
            "method_preference": cognitive_state["preferred_method"],
            "recent_success_rate": cognitive_state["recent_success_rate"]
        }

        # 分析粗心程度趋势
        if len(self.student_agent.cognitive_state.problem_solving_history) >= 3:
            recent_errors = sum(1 for record in self.student_agent.cognitive_state.problem_solving_history[-3:]
                                if record["errors"])
            earlier_errors = sum(1 for record in self.student_agent.cognitive_state.problem_solving_history[-6:-3]
                                 if record["errors"])

            if recent_errors < earlier_errors:
                progress_info["carelessness_trend"] = "improving"
            elif recent_errors > earlier_errors:
                progress_info["carelessness_trend"] = "worsening"

        return progress_info

    def _setup_conversation_summary(self):
        """配置对话摘要功能"""
        summary_enabled = self.config.agent.get("conversation_summary", False)
        if not summary_enabled:
            self.summary_llm = None
            return

        try:
            # 使用专门配置的摘要模型，如果没有配置则使用轻量级模型
            summary_model_name = self.config.agent.get("summary_model_name", "qwen2.5:0.5b")
            summary_model_type = self.config.agent.get("summary_model_type", "local")
            summary_base_url = self.config.agent.get("summary_base_url", "http://localhost:11434")

            if summary_model_type == "local":
                from langchain_ollama import ChatOllama
                self.summary_llm = ChatOllama(
                    model=summary_model_name,
                    base_url=summary_base_url,
                    temperature=self.config.agent.get("temperature_summary_model", 0.2),
                    num_predict=self.config.agent.get("max_tokens_summary_model", 2000),
                )
                logger.info(f"✅ Summary model: {summary_model_name} (local)")
            else:
                # API摘要模型
                api_kwargs = {
                    "temperature": 0.1,
                    "timeout": 20,
                    "max_tokens": 300,
                }
                self.summary_llm = init_chat_model(summary_model_name, **api_kwargs)
                logger.info(f"✅ Summary model: {summary_model_name} (API)")

        except Exception as e:
            logger.warning(f"❌ Failed to setup summary model: {e}")
            logger.info("⚠️  Summary feature will use main model if available")
            self.summary_llm = None

    def get_memory_statistics(self):
        """获取内存统计信息"""
        stats = {}
        if self.student_agent:
            stats["student"] = self.student_agent.get_memory_stats()
        if self.teacher_agent:
            stats["teacher"] = self.teacher_agent.get_memory_stats()
        return stats

    def clear_all_memory(self):
        """清空所有内存"""
        if self.student_agent:
            self.student_agent.clear_memory()
        if self.teacher_agent:
            self.teacher_agent.clear_memory()

    def visualize_results(self, output_dir: str = "results"):
        """可视化实验结果"""
        if not self.experiment_recorder.records:
            print("❌ 没有实验数据可供可视化")
            return

        # 计算统计信息
        self.experiment_recorder.calculate_statistics()
        stats = self.experiment_recorder.summary_stats

        # 设置绘图风格
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multi-Agent Math Tutoring Experiment Results', fontsize=16, fontweight='bold')

        # 1. 准确率和答案泄露率
        metrics = ['Accuracy', 'Answer Leakage Rate']
        values = [stats['accuracy'], stats['answer_leakage_rate']]
        colors = ['#2ecc71', '#e74c3c']

        bars1 = axes[0, 0].bar(metrics, values, color=colors, alpha=0.7)
        axes[0, 0].set_title('Accuracy vs Answer Leakage')
        axes[0, 0].set_ylabel('Rate')
        axes[0, 0].set_ylim(0, 1)

        # 在柱状图上添加数值标签
        for bar, value in zip(bars1, values):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # 2. 平均对话轮数分布
        turn_counts = [r.total_turns for r in self.experiment_recorder.records]
        axes[0, 1].hist(turn_counts, bins=range(1, max(turn_counts) + 2), alpha=0.7, color='#3498db', edgecolor='black')
        axes[0, 1].set_title('Distribution of Dialogue Turns')
        axes[0, 1].set_xlabel('Number of Turns')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(stats['avg_turns_per_problem'], color='red', linestyle='--',
                           label=f'Average: {stats["avg_turns_per_problem"]:.2f}')
        axes[0, 1].legend()

        # 3. 并行思考和思考路径
        thinking_data = ['Parallel Thinking', 'Thinking Paths']
        thinking_values = [stats['avg_parallel_thinking'], stats['avg_thinking_paths']]

        bars2 = axes[0, 2].bar(thinking_data, thinking_values, color=['#9b59b6', '#f39c12'], alpha=0.7)
        axes[0, 2].set_title('Average Thinking Metrics per Problem')
        axes[0, 2].set_ylabel('Average Count')

        for bar, value in zip(bars2, thinking_values):
            axes[0, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                            f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

        # 4. 教师意图分布
        teacher_intents = []
        for record in self.experiment_recorder.records:
            for turn in record.turns:
                if 'teacher_intent' in turn:
                    teacher_intents.append(turn['teacher_intent'])

        if teacher_intents:
            intent_counts = pd.Series(teacher_intents).value_counts()
            axes[1, 0].pie(intent_counts.values, labels=intent_counts.index, autopct='%1.1f%%',
                           startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, len(intent_counts))))
            axes[1, 0].set_title('Teacher Response Intent Distribution')

        # 5. 正确答案 vs 错误答案
        correct_data = ['Correct', 'Incorrect']
        correct_values = [stats['correct_answers'], stats['total_problems'] - stats['correct_answers']]

        bars3 = axes[1, 1].bar(correct_data, correct_values, color=['#27ae60', '#c0392b'], alpha=0.7)
        axes[1, 1].set_title('Correct vs Incorrect Answers')
        axes[1, 1].set_ylabel('Count')

        for bar, value in zip(bars3, correct_values):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                            f'{value}', ha='center', va='bottom', fontweight='bold')

        # 6. 轮次与准确率的关系
        turns_vs_correct = []
        for record in self.experiment_recorder.records:
            turns_vs_correct.append((record.total_turns, 1 if record.correct else 0))

        if turns_vs_correct:
            df = pd.DataFrame(turns_vs_correct, columns=['turns', 'correct'])
            accuracy_by_turns = df.groupby('turns')['correct'].mean().reset_index()
            axes[1, 2].plot(accuracy_by_turns['turns'], accuracy_by_turns['correct'],
                            marker='o', linewidth=2, markersize=8, color='#e67e22')
            axes[1, 2].set_title('Accuracy by Number of Turns')
            axes[1, 2].set_xlabel('Number of Turns')
            axes[1, 2].set_ylabel('Accuracy Rate')
            axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()

        # 保存图片
        import os
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir,
                                 f"{self.experiment_recorder.experiment_name}_{self.experiment_recorder.timestamp}_plots.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ 可视化结果已保存到: {plot_file}")
        return plot_file

    def visualize_cognitive_progress(self, output_dir: str = "results"):
        """可视化认知状态进步"""
        if not self.config.agent.get("use_cognitive_state", False):
            return None

        if not self.student_agent or not self.student_agent.cognitive_state:
            return None

        # 收集历史数据
        history = self.student_agent.cognitive_state.problem_solving_history
        if len(history) < 2:
            return None

        # 创建认知状态变化图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Student Cognitive State Progress', fontsize=16, fontweight='bold')

        # 1. 粗心程度变化
        carelessness_levels = []
        success_rates = []

        # 计算滑动窗口的成功率
        window_size = min(3, len(history))
        for i in range(len(history)):
            window_start = max(0, i - window_size + 1)
            window = history[window_start:i + 1]
            success_rate = sum(1 for record in window if record["success"]) / len(window)
            success_rates.append(success_rate)

            # 模拟粗心程度变化（基于错误频率）
            recent_errors = sum(1 for record in window if record["errors"])
            carelessness = max(1, min(10, 5 + recent_errors * 2))  # 简单模拟
            carelessness_levels.append(carelessness)

        # 粗心程度图
        axes[0, 0].plot(range(len(carelessness_levels)), carelessness_levels,
                        marker='o', linewidth=2, color='#e74c3c')
        axes[0, 0].set_title('Carelessness Level Trend')
        axes[0, 0].set_xlabel('Problem Sequence')
        axes[0, 0].set_ylabel('Carelessness Level (1-10)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(1, 10)

        # 2. 成功率变化
        axes[0, 1].plot(range(len(success_rates)), success_rates,
                        marker='s', linewidth=2, color='#2ecc71')
        axes[0, 1].set_title('Success Rate Trend')
        axes[0, 1].set_xlabel('Problem Sequence')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 1)

        # 3. 方法偏好分布
        method_counts = {}
        for record in history:
            method = record.get("method_used", "unknown")
            method_counts[method] = method_counts.get(method, 0) + 1

        if method_counts:
            methods = list(method_counts.keys())
            counts = list(method_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

            axes[1, 0].pie(counts, labels=methods, autopct='%1.1f%%',
                           colors=colors, startangle=90)
            axes[1, 0].set_title('Method Preference Distribution')

        # 4. 错误类型分析
        error_types = {}
        for record in history:
            for error in record.get("errors", []):
                error_types[error] = error_types.get(error, 0) + 1

        if error_types:
            errors = list(error_types.keys())
            error_counts = list(error_types.values())

            bars = axes[1, 1].bar(range(len(errors)), error_counts,
                                  color='#f39c12', alpha=0.7)
            axes[1, 1].set_title('Error Type Analysis')
            axes[1, 1].set_xlabel('Error Type')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_xticks(range(len(errors)))
            axes[1, 1].set_xticklabels(errors, rotation=45)

            # 添加数值标签
            for bar, count in zip(bars, error_counts):
                axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                                f'{count}', ha='center', va='bottom')

        plt.tight_layout()

        # 保存图片
        import os
        os.makedirs(output_dir, exist_ok=True)
        plot_file = os.path.join(output_dir, f"cognitive_progress_{self.experiment_recorder.timestamp}.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Cognitive progress visualization saved to: {plot_file}")
        return plot_file


def create_raw_dataset(data_paths, data_config):
    # 保持原有实现
    dataset = RawDataset(parquet_file=data_paths, config=data_config)
    return dataset


def solve_with_dialogue(config, logger, val_dataset, multi_agent_system):
    """使用多智能体对话解决问题"""
    first_correct = torch.zeros(1, dtype=torch.float32, device=device_name)
    correct = torch.zeros(1, dtype=torch.float32, device=device_name)
    dialogue_count = torch.zeros(1, dtype=torch.float32, device=device_name)
    parallel_thinking_count = torch.zeros(1, dtype=torch.float32, device=device_name)
    thinking_paths_count = torch.zeros(1, dtype=torch.float32, device=device_name)
    leaked_answer_count = torch.zeros(1, dtype=torch.float32, device=device_name)
    total = torch.zeros(1, dtype=torch.float32, device=device_name)
    error_ans = torch.zeros(1, dtype=torch.float32, device=device_name)
    # 新指标：一题多解能力
    multi_solution_scores = []
    cognitive_progress = []

    student_agent, teacher_agent = multi_agent_system.initialize_agents()

    explicit_mode = config.agent.get("explicit_interaction", True)
    eval_mode = 'Explicit Interaction' if explicit_mode else 'Tool-based'

    if config.agent.get("max_samples", None):
        max_samples = config.agent.get("max_samples", min(10, len(val_dataset.prompts)))
    else:
        max_samples = len(val_dataset.prompts)

    desc = f"Multi-agent Evaluation [{eval_mode}] - Dataset: {'GSM8k' if 'gsm' in config.data.val_files.lower() else 'MATH'}"

    for i, (raw_problem, prompt, label_ans) in tqdm(
            enumerate(zip(val_dataset.raw_problem[:max_samples], val_dataset.prompts[:max_samples], val_dataset.responses[:max_samples])),
            desc=desc, total=max_samples
    ):
        # 在对话开始前清空内存（确保每次问题独立）
        multi_agent_system.clear_all_memory()

        # 生成解题树（如果启用）
        if config.agent.get("use_solution_tree", False):
            solution_tree = multi_agent_system.generate_solution_tree(raw_problem)
            if solution_tree:
                logger.info(f"🌳 Generated solution tree with {len(solution_tree.solution_paths)} expert paths")

        # 创建对话记录
        dialogue_record = DialogueRecord(prompt, label_ans,
                                         debug_mode=True if i < config.agent.debug_samples else False)

        # 记录学生初始认知状态
        initial_cognitive_state = multi_agent_system.get_cognitive_state()

        if explicit_mode:
            # 模式1: 显式交互
            final_answer, correct_answer, dialogue_record = student_agent.multi_agent_chat_explicit(
                teacher_agent, prompt, raw_problem, label_ans, dialogue_record
            )
        else:
            # 模式2: 工具调用
            correct_answer = None
            final_answer = student_agent.multi_agent_chat_tool_based(
                prompt, label_ans, dialogue_record
            )

        total += 1

        if not final_answer:
            error_ans += 1
            dialogue_record.correct = False
            success = False
        else:
            if dialogue_record.first_correct:
                first_correct += 1
            if dialogue_record.correct:
                correct += 1
            success = dialogue_record.correct

        # 添加记录到实验记录器
        multi_agent_system.experiment_recorder.add_record(dialogue_record)

        if config.agent.get("use_cognitive_state", False):
            multi_agent_system.update_cognitive_state_based_on_dialogue(
                prompt, dialogue_record, success
            )

            # 记录认知状态变化
            current_state = multi_agent_system.get_cognitive_state()
            if initial_cognitive_state and current_state:
                logger.info(
                    f"🧠 Cognitive State Updated - Carelessness: {initial_cognitive_state['carelessness_level']} -> {current_state['carelessness_level']}")

                # 分析进步情况
                progress = multi_agent_system.analyze_student_progress(dialogue_record)
                if progress:
                    cognitive_progress.append(progress)
                    logger.info(
                        f"📈 Progress: {progress['carelessness_trend']}, Success Rate: {progress['recent_success_rate']:.2f}")

        # 计算一题多解分数（如果启用解题树）
        # 在对话结束后，输出解题树信息
        if config.agent.get("use_solution_tree", False) and multi_agent_system.current_solution_tree:
            solution_tree = multi_agent_system.current_solution_tree
            logger.info(f"🌳 解题树统计:")
            logger.info(f"   专家路径数: {len([p for p in solution_tree.solution_paths if p.get('type') == 'expert'])}")
            logger.info(
                f"   学生路径数: {len([p for p in solution_tree.solution_paths if p.get('type') == 'student'])}")

            # 输出学生路径详情
            student_paths = [p for p in solution_tree.solution_paths if p.get('type') == 'student']
            for i, path in enumerate(student_paths):
                logger.info(
                    f"   学生路径 {i + 1}: 步骤数={len(path.get('steps', []))}, 成功={path.get('success', False)}, 方法={path.get('method', 'unknown')}")
            multi_solution_score = _calculate_multi_solution_score(multi_agent_system.current_solution_tree)
            multi_solution_scores.append(multi_solution_score)
            logger.info(f"🔢 Multi-solution Score: {multi_solution_score:.2f}")

        dialogue_count += dialogue_record.total_turns
        parallel_thinking_count += dialogue_record.parallel_thinking_count
        thinking_paths_count += dialogue_record.thinking_paths_count
        leaked_answer_count += dialogue_record.leaked_answer

        # 打印当前进度
        current_first_accuracy = first_correct.item() / total.item() if total.item() > 0 else 0
        current_accuracy = correct.item() / total.item() if total.item() > 0 else 0
        avg_dialogue_count = dialogue_count.item() / total.item() if total.item() > 0 else 0
        avg_parallel_thinking_count = parallel_thinking_count.item() / total.item() if total.item() > 0 else 0
        avg_thinking_paths_count = thinking_paths_count.item() / total.item() if total.item() > 0 else 0
        avg_leaked_answer_count = leaked_answer_count.item() / total.item() if total.item() > 0 else 0

        logger.info(f'\n📊 Sample {i + 1}/{max_samples}:')
        logger.info(f'   First Correct: {dialogue_record.first_correct}')
        logger.info(f'   Final Correct: {dialogue_record.correct}')
        logger.info(f'   Current First Accuracy: {current_first_accuracy:.4f}')
        logger.info(f'   Current Final Accuracy: {current_accuracy:.4f}')
        logger.info(f'   Current Avg Dialogue Round: {avg_dialogue_count:.4f}')
        logger.info(f'   Current Avg Parallel Thinking Count: {avg_parallel_thinking_count:.4f}')
        logger.info(f'   Current Avg Thinking Paths Count: {avg_thinking_paths_count:.4f}')
        logger.info(f'   Current Avg Leaked Answer: {avg_leaked_answer_count:.4}')

        # 每处理完一个问题后打印内存统计
        if config.agent.get("conversation_summary", False):
            memory_stats = multi_agent_system.get_memory_statistics()
            logger.info(f"🧠 内存统计 - 学生: {memory_stats['student']}, 教师: {memory_stats['teacher']}")

    # 计算最终统计并保存结果
    multi_agent_system.experiment_recorder.calculate_statistics()
    multi_agent_system.experiment_recorder.save_results()
    multi_agent_system.experiment_recorder.print_summary()

    # 计算一题多解平均分
    if multi_solution_scores:
        avg_multi_solution = sum(multi_solution_scores) / len(multi_solution_scores)
        logger.info(f"\n🎯 Average Multi-Solution Score: {avg_multi_solution:.3f}")

    # 分析认知进步趋势
    if cognitive_progress:
        improving_count = sum(1 for progress in cognitive_progress if progress["carelessness_trend"] == "improving")
        improvement_rate = improving_count / len(cognitive_progress)
        logger.info(f"🧠 Cognitive Improvement Rate: {improvement_rate:.3f} ({improving_count}/{len(cognitive_progress)})")

    # 生成可视化
    multi_agent_system.visualize_results()
    multi_agent_system.visualize_cognitive_progress()

    final_accuracy = correct.item() / total.item() if total.item() > 0 else 0
    logger.info(f"\n🎯 Final Results:")
    logger.info(f"   Total Samples: {total.item()}")
    logger.info(f"   Correct Answers: {correct.item()}")
    logger.info(f"   Error Answers: {error_ans.item()}")
    logger.info(f"   Final Accuracy: {final_accuracy:.4f}")

    return final_accuracy


def _calculate_multi_solution_score(solution_tree):
    """计算一题多解分数"""
    if not solution_tree or not solution_tree.solution_paths:
        return 0.0

    expert_paths = [p for p in solution_tree.solution_paths if p["type"] == "expert"]
    student_paths = [p for p in solution_tree.solution_paths if p["type"] == "student"]

    if not expert_paths:
        return 0.0

    # 基于专家路径数量和学生尝试的方法多样性评分
    base_score = min(len(expert_paths) / 3.0, 1.0)  # 最多3个专家路径

    if student_paths:
        student_methods = set(p["method"] for p in student_paths if p["method"] != "unknown")
        method_diversity = len(student_methods) / len(expert_paths)
        return (base_score + method_diversity) / 2.0
    else:
        return base_score * 0.5  # 没有学生路径时分数减半


def _analyze_student_approach(dialogue_record):
    """分析学生解题方法"""
    if not dialogue_record.turns:
        return "unknown"

    # 从对话记录中提取学生回复
    student_responses = []
    for turn in dialogue_record.turns:
        if 'student_response' in turn and turn['student_response']:
            student_responses.append(turn['student_response'])

    if not student_responses:
        return "unknown"

    # 分析最后的学生回复
    last_response = student_responses[-1].lower()

    # 检测方法类型
    if any(word in last_response for word in ["equation", "solve for", "variable", "x =", "let x"]):
        return "algebraic"
    elif any(word in last_response for word in ["diagram", "graph", "shape", "angle", "area", "triangle"]):
        return "geometric"
    elif any(word in last_response for word in ["calculate", "compute", "number", "digit", "sum", "total"]):
        return "computational"
    elif any(word in last_response for word in ["logic", "reason", "therefore", "because", "since"]):
        return "logical"
    elif any(word in last_response for word in ["guess", "try", "maybe", "perhaps"]):
        return "trial_and_error"
    else:
        return "general"


def _extract_errors(dialogue_record):
    """从对话记录中提取错误模式"""
    errors = []

    if not dialogue_record.turns:
        return errors

    # 分析教师回复中的纠正信息
    for turn in dialogue_record.turns:
        if 'teacher_response' in turn and turn['teacher_response']:
            teacher_response = turn['teacher_response'].lower()

            # 检测错误类型
            if any(word in teacher_response for word in ["wrong", "incorrect", "mistake", "error"]):
                if "calculation" in teacher_response or "compute" in teacher_response:
                    errors.append("calculation_error")
                elif "concept" in teacher_response or "understand" in teacher_response:
                    errors.append("conceptual_error")
                elif "method" in teacher_response or "approach" in teacher_response:
                    errors.append("methodological_error")
                elif "step" in teacher_response or "process" in teacher_response:
                    errors.append("procedural_error")
                else:
                    errors.append("general_error")

    return errors


def _detect_method_used(dialogue_record):
    """检测学生使用的主要方法"""
    if not dialogue_record.turns:
        return "unknown"

    # 收集所有学生回复
    all_student_text = ""
    for turn in dialogue_record.turns:
        if 'student_response' in turn and turn['student_response']:
            all_student_text += " " + turn['student_response'].lower()

    # 方法检测
    method_scores = {
        "algebraic": 0,
        "geometric": 0,
        "computational": 0,
        "logical": 0
    }

    # 关键词匹配
    algebraic_keywords = ["equation", "variable", "solve for", "x =", "let x", "algebra"]
    geometric_keywords = ["diagram", "graph", "shape", "angle", "area", "triangle", "circle"]
    computational_keywords = ["calculate", "compute", "number", "digit", "sum", "total", "multiply"]
    logical_keywords = ["logic", "reason", "therefore", "because", "since", "if then"]

    for keyword in algebraic_keywords:
        if keyword in all_student_text:
            method_scores["algebraic"] += 1

    for keyword in geometric_keywords:
        if keyword in all_student_text:
            method_scores["geometric"] += 1

    for keyword in computational_keywords:
        if keyword in all_student_text:
            method_scores["computational"] += 1

    for keyword in logical_keywords:
        if keyword in all_student_text:
            method_scores["logical"] += 1

    # 返回得分最高的方法
    if not any(method_scores.values()):
        return "unknown"

    return max(method_scores.items(), key=lambda x: x[1])[0]


def _analyze_student_response_characteristics(dialogue_record):
    """分析学生回复特征"""
    if not dialogue_record.turns:
        return "neutral"

    student_responses = []
    for turn in dialogue_record.turns:
        if 'student_response' in turn and turn['student_response']:
            student_responses.append(turn['student_response'])

    if not student_responses:
        return "neutral"

    # 分析回复长度和内容特征
    total_length = sum(len(response) for response in student_responses)
    avg_length = total_length / len(student_responses)

    last_response = student_responses[-1].lower()

    # 判断回复风格
    if avg_length > 300:
        return "detailed"
    elif avg_length < 100:
        return "brief"
    elif any(word in last_response for word in ["i think", "maybe", "perhaps", "not sure"]):
        return "thoughtful"
    elif any(word in last_response for word in ["obviously", "clearly", "definitely"]):
        return "confident"
    else:
        return "neutral"



def main(config, logger):
    val_dataset = create_raw_dataset(config.data.val_files, config.data)
    multi_agent_system = MultiAgentSystem(config)

    accuracy = solve_with_dialogue(config, logger, val_dataset, multi_agent_system)
    logger.info(f"\n🎯 Final Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    import hydra
    from omegaconf import OmegaConf
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    hydra.initialize(config_path="utils", version_base=None)

    overrides = []

    ''' 
    -----------------------------------
    ---------- 系统配置和超参数 ----------
    -----------------------------------
    '''
    overrides.append('data.val_files=D:\DeepLearning\Code\LangChain\dataset/APO_combine_with_source_test_without_path_2.parquet')
    # overrides.append('data.val_files=/mnt/t2-6tb/medical/SocraticLM_langchain/LangChain_3090/dataset/GSM8k_test_with_prompt4.parquet')
    overrides.append('+model.log_folder_path=D:\DeepLearning\Code\LangChain\log')
    # overrides.append('+model.log_folder_path=/mnt/t2-6tb/medical/SocraticLM_langchain/LangChain_3090/log')
    overrides.append('data.prompt_key=extra_info')
    overrides.append('data.response_key=extra_info')
    overrides.append('+data.raw_problem_key=extra_info')
    overrides.append('+data.first_prompt=raw')
    overrides.append('+data.prompt_dict_keys=[question]')
    overrides.append('+data.response_dict_keys=[answer]')
    overrides.append('+data.raw_problem_dict_keys=[raw_problem]')
    # overrides.append('+agent.max_samples=200')    # 测试用样本数量
    overrides.append('+agent.debug_samples=10')   # 查看完整response
    overrides.append('+agent.debug_mode=false')  # 查看完整prompt
    overrides.append('+agent.max_turns=3')

    ''' 
    -----------------------------------
    ---------- 智能体与对话配置 ----------
    -----------------------------------
    '''
    overrides.append('+model.api_name_student=deepseek-chat')
    overrides.append('+model.model_type_student=api')
    # overrides.append('+model.model_type_student=local')
    # overrides.append('+model.api_name_student=qwen3-4b-4k:latest')  # qwen3-4b-4k:latest   qwen3:4b-tuned-4k
    overrides.append('+model.base_url_student=http://localhost:11434')
    overrides.append('+model.temperature_student=0')
    overrides.append('+model.max_tokens_student=2000')

    overrides.append('+model.api_name_teacher=deepseek-chat')
    overrides.append('+model.model_type_teacher=api')
    # overrides.append('+model.model_type_teacher=local')
    # overrides.append('+model.api_name_teacher=qwen3-4b-4k:latest')  # qwen3-4b-4k:latest   qwen3:4b-tuned-4k
    overrides.append('+model.base_url_teacher=http://localhost:11434')
    overrides.append('+model.temperature_teacher=0')
    overrides.append('+model.max_tokens_teacher=2000')

    ''' 
    -----------------------------------
    ------------ 解题树配置 -------------
    -----------------------------------
    '''
    overrides.append('+agent.use_solution_tree=false')                 # 启用解题树
    overrides.append('+agent.evaluate_multi_solution=false')           # 评估一题多解能力
    overrides.append('+model.api_name_expert=deepseek-chat')
    overrides.append('+model.model_type_expert=api')
    # overrides.append('+model.model_type_expert=local')
    # overrides.append('+model.api_name_expert=qwen3-4b-4k:latest')  # qwen3-4b-4k:latest   qwen3:4b-tuned-4k
    overrides.append('+model.base_url_expert=http://localhost:11434')
    overrides.append('+model.temperature_expert=0')
    overrides.append('+model.max_tokens_expert=2000')

    ''' 
    -----------------------------------
    ----------- 对话摘要配置 ------------
    -----------------------------------
    '''
    overrides.append('+agent.conversation_summary=false')  # 启用对话摘要
    overrides.append('+agent.summary_model_name=deepseek-chat')
    overrides.append('+agent.summary_model_type=api')
    # overrides.append('+agent.summary_model_type=local')
    # overrides.append('+agent.summary_model_name=qwen3-4b-4k:latest')  # qwen3-4b-4k:latest   qwen3:4b-tuned-4k
    overrides.append('+agent.summary_base_url=http://localhost:11434')
    overrides.append('+model.temperature_summary_model=0')
    overrides.append('+model.max_tokens_summary_model=2000')
    overrides.append('+agent.summary_max_turns=8')  # 每8轮对话生成摘要
    overrides.append('+agent.summary_max_tokens=1500')  # 最大token限制
    overrides.append('+agent.explicit_interaction=true')  # true=显式交互, false=工具调用

    ''' 
    -----------------------------------
    ---------- Agent个性化配置 ----------
    -----------------------------------
    '''
    overrides.append('+agent.parallel_thinking=false')                 # 学生并行思考能力
    overrides.append('+agent.socratic_teaching=false')                 # 教师苏格拉底教学
    overrides.append('+agent.student_sys_prompt=STUDENT_PROMPT_EASY')  # 选择学生系统提示词
    overrides.append('+agent.teacher_sys_prompt=TEACHER_PROMPT_EASY')  # 选择教师系统提示词
    overrides.append('+agent.teacher_tree_sys_prompt=TEACHER_WITH_TREE_PROMPT')  # 选择解题树系统提示词
    overrides.append('+agent.student_tree_sys_prompt=STUDENT_WITH_TREE_PROMPT')  # 选择解题树系统提示词
    overrides.append('+agent.expert_sys_prompt=EXPERT_STUDENT_PROMPT')  # 选择教师系统提示词
    overrides.append('+agent.use_cognitive_state=true')               # 学生认知状态配置 启用认知状态
    overrides.append('+agent.carelessness_level=9')                    # 粗心程度 (1-10)
    overrides.append('+agent.math_background=beginner')                    # 数学背景
    overrides.append('+agent.response_style=brief')               # 回复风格
    overrides.append('+agent.preferred_method=algebraic')              # 偏好方法
    overrides.append('+agent.learning_style=reading-writing')                   # 学习风格

    config = hydra.compose(config_name="sft_trainer", overrides=overrides)
    get_logger(config)
    logger.info(OmegaConf.to_yaml(config))
    main(config, logger)