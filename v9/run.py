from function.agent_IO import ExperimentRecorder, ExpertStudentAgent, SimpleAgent, SummaryConfig, StudentCognitiveState, DialogueRecord
from utils.evaluator import *
from utils.dataset.parallel_thinking_sft_dataset import RawDataset
from config_secret.api_config import ds_key_config, ds_key_url, yi_key_config, yi_key_url, smith_key
from v9.function.api_client import create_api_client
from v9.function.middleware import ModelConfig
from tqdm import tqdm
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import graphviz

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
            api_key=self.config.model.api_student_key,
            temperature=self.config.model.temperature_student,
            max_tokens=self.config.model.max_tokens_student
        )

        # 教师模型配置
        teacher_model_config = ModelConfig(
            model_type=self.config.model.model_type_teacher,
            model_name=self.config.model.api_name_teacher,
            base_url=self.config.model.base_url_teacher if hasattr(self.config.model, 'base_url_teacher') else None,
            api_key=self.config.model.api_teacher_key,
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
                base_url=self.config.model.base_url_expert if hasattr(self.config.model, 'base_url_teacher') else None,
                api_key=self.config.model.api_key_expert,
                temperature=self.config.model.temperature_expert,
                max_tokens=self.config.model.max_tokens_expert
            )
            self.expert_agent.model_init(expert_model_config, 'expert')
            self.expert_agent.config_create("thread_id", "expert_1")
            self.expert_agent.agent_init(
                expert_model_config,
                prompt_sys_name=self.config.agent.expert_sys_prompt,
            )
            logger.info(f"✅ Expert agent initialized: {self.config.model.api_name_expert}")

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
            student_response=self.config.agent.student_response,
            **student_kwargs
        )
        self.teacher_agent.agent_init(
            teacher_model_config,
            prompt_sys_name=self.config.agent.teacher_sys_prompt,
            teacher_response=self.config.agent.teacher_response,
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
            self.student_agent.enable_conversation_summary(summary_config, self.summary_llm,
                                                           summary_mode=self.config.agent.summary_mode)
            self.teacher_agent.enable_conversation_summary(summary_config, self.summary_llm,
                                                           summary_mode=self.config.agent.summary_mode)
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

    def generate_solution_tree(self, problem, answer):
        """生成解题树"""
        if self.expert_agent and hasattr(self.expert_agent, 'generate_solution_tree'):
            self.current_solution_tree, solution_tree_section = self.expert_agent.generate_solution_tree(problem, answer)

            # 设置解题树到学生和教师agent
            if self.current_solution_tree:
                self.student_agent.set_solution_tree(self.current_solution_tree, solution_tree_section)
                self.teacher_agent.set_solution_tree(self.current_solution_tree, solution_tree_section)

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

    def analyze_student_progress(self, dialogue_record):
        """分析单次对话中学生的进步情况"""
        if not self.config.agent.get("use_cognitive_state", False):
            return None

        if not dialogue_record or not dialogue_record.turns:
            return None

        progress_info = {
            "turn_progress": [],
            "error_reduction": "stable",
            "method_improvement": "stable",
            "response_quality": "stable",
            "final_success": dialogue_record.correct
        }

        # 分析每轮对话的进步情况
        turns = dialogue_record.turns
        for i, turn in enumerate(turns):
            turn_info = {
                "turn": turn.get("turn", i + 1),
                "errors": [],
                "method_used": "",
                "response_quality": "neutral"
            }

            # 提取错误信息
            if 'teacher_response' in turn:
                turn_info["errors"] = self._extract_turn_errors(turn['teacher_response'])

            # 分析解题方法
            if 'student_response' in turn:
                turn_info["method_used"] = self._detect_turn_method(turn['student_response'])
                turn_info["response_quality"] = self._analyze_turn_response_quality(turn['student_response'])

            progress_info["turn_progress"].append(turn_info)

        # 分析整体趋势
        if len(turns) >= 2:
            # 错误趋势分析
            early_errors = sum([len(t["errors"]) for t in progress_info["turn_progress"][:len(turns)//2]])
            later_errors = sum([len(t["errors"]) for t in progress_info["turn_progress"][len(turns)//2:]])
            if later_errors < early_errors:
                progress_info["error_reduction"] = "improving"
            elif later_errors > early_errors:
                progress_info["error_reduction"] = "worsening"

            # 方法改进分析
            methods_used = [t["method_used"] for t in progress_info["turn_progress"] if t["method_used"]]
            if len(set(methods_used)) > 1:
                progress_info["method_improvement"] = "exploring"

            # 回复质量分析
            quality_scores = {"neutral": 0, "improved": 1, "declined": -1}
            early_quality = sum(quality_scores[t["response_quality"]] for t in progress_info["turn_progress"][:len(turns)//2])
            later_quality = sum(quality_scores[t["response_quality"]] for t in progress_info["turn_progress"][len(turns)//2:])
            if later_quality > early_quality:
                progress_info["response_quality"] = "improving"
            elif later_quality < early_quality:
                progress_info["response_quality"] = "declining"

        return progress_info

    def _extract_turn_errors(self, teacher_response):  # dddd
        """分析单轮教师反馈中的错误"""
        errors = []
        if not teacher_response:
            return errors

        response_lower = teacher_response.lower()
        error_patterns = {
            "calculation_error": ["calculation", "compute", "arithmetic"],
            "conceptual_error": ["concept", "understand", "principle"],
            "methodological_error": ["method", "approach", "strategy"],
            "logical_error": ["logic", "reasoning", "conclusion"]
        }

        for error_type, keywords in error_patterns.items():
            if any(word in response_lower for word in keywords):
                errors.append(error_type)

        return errors

    def _detect_turn_method(self, student_response):
        """分析单轮学生使用的解题方法"""
        if not student_response:
            return ""

        response_lower = student_response.lower()
        method_patterns = {
            "algebraic": ["equation", "variable", "solve for", "x ="],
            "geometric": ["diagram", "graph", "shape", "angle"],
            "computational": ["calculate", "compute", "sum", "multiply"],
            "logical": ["reason", "because", "therefore", "since"]
        }

        for method, keywords in method_patterns.items():
            if any(word in response_lower for word in keywords):
                return method

        return "general"

    def _analyze_turn_response_quality(self, student_response):
        """分析单轮回复质量"""
        if not student_response:
            return "neutral"

        response_lower = student_response.lower()
        
        # 改进指标
        improvement_indicators = [
            "therefore", "because", "let me explain",
            "first", "second", "finally",
            "i understand now", "alternatively"
        ]
        
        # 退步指标
        decline_indicators = [
            "i'm not sure", "maybe", "i guess",
            "i don't know", "this is difficult"
        ]

        if any(indicator in response_lower for indicator in improvement_indicators):
            return "improved"
        elif any(indicator in response_lower for indicator in decline_indicators):
            return "declined"
            
        return "neutral"

    def _analyze_error_patterns(self, error_patterns):
        """分析常见错误模式"""
        if not error_patterns:
            return {}
            
        # 统计错误类型频率
        error_counts = {}
        for error in error_patterns:
            error_counts[error] = error_counts.get(error, 0) + 1
            
        # 返回频率最高的3种错误
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_errors[:3])

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

            # 创建摘要模型配置
            summary_model_config = ModelConfig(
                model_type=self.config.agent.summary_model_type,
                model_name=self.config.agent.summary_model_name,
                base_url=self.config.agent.summary_base_url,
                api_key=self.config.agent.summary_api_key,
                temperature=self.config.model.temperature_summary_model,
                max_tokens=self.config.model.max_tokens_summary_model
            )
            
            # 使用原始API客户端
            self.summary_llm = create_api_client(summary_model_config)
            logger.info(f"✅ Summary model: {summary_model_name} ({summary_model_type})")

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

        # 使用已更新的认知状态数据
        cognitive_state = self.student_agent.cognitive_state
        history = cognitive_state.problem_solving_history
        
        # 确保输出目录存在
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            import matplotlib.pyplot as plt
            import pandas as pd
            from datetime import datetime
            
            # 转换历史数据为DataFrame
            df = pd.DataFrame(history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # 计算滑动平均成功率
            df['success_rate'] = df['success'].rolling(window=5, min_periods=1).mean()
            
            # 创建图表
            plt.figure(figsize=(12, 8))
            
            # 绘制成功率趋势
            plt.subplot(2, 1, 1)
            plt.plot(range(len(df)), df['success_rate'], marker='o')
            plt.title('Student Learning Progress')
            plt.xlabel('Problem Number')
            plt.ylabel('Success Rate (5-problem moving average)')
            plt.grid(True)
            
            # 统计方法使用情况
            method_counts = {}
            for method in df['method_used'].dropna():
                method_counts[method] = method_counts.get(method, 0) + 1
            
            # 绘制方法使用分布
            if method_counts:
                plt.subplot(2, 1, 2)
                plt.bar(method_counts.keys(), method_counts.values())
                plt.title('Problem-Solving Methods Distribution')
                plt.xlabel('Method')
                plt.ylabel('Frequency')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'cognitive_progress.png'))
            plt.close()
            
            # 保存统计数据
            stats = {
                'carelessness_level': cognitive_state.carelessness_level,
                'math_background': cognitive_state.math_background,
                'preferred_method': cognitive_state.preferred_method,
                'average_success_rate': df['success'].mean(),
                'total_problems': len(df),
                'method_distribution': method_counts,
                'error_patterns': cognitive_state.error_patterns
            }
            
            with open(os.path.join(output_dir, 'cognitive_stats.json'), 'w') as f:
                json.dump(stats, f, indent=2)
                
            logger.info("✅ Cognitive progress visualization saved")
            
        except Exception as e:
            logger.error(f"❌ Error creating cognitive progress visualization: {e}")
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
        solution_tree = None
        if config.agent.get("use_solution_tree", False):
            solution_tree = multi_agent_system.generate_solution_tree(raw_problem, extract_answer(label_ans))
            if solution_tree:
                logger.info(f"🌳 Generated solution tree by expert.")

        # 创建对话记录
        dialogue_record = DialogueRecord(prompt, label_ans,
                                         debug_mode=True if i < config.agent.debug_samples else False)

        # 记录学生初始认知状态 # dddd
        initial_cognitive_state = None
        if config.agent.get("use_cognitive_state", False):
            initial_cognitive_state = multi_agent_system.get_cognitive_state()

        if explicit_mode:
            # 模式1: 显式交互
            if solution_tree:
                final_answer, correct_answer, dialogue_record = student_agent.multi_agent_chat_explicit(
                    teacher_agent, prompt, raw_problem, label_ans, dialogue_record, solution_tree=solution_tree
                )
            else:
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

        # 分析单次对话中的进步情况并记录认知进步趋势
        if config.agent.get("use_cognitive_state", False):
            # 只调用一次分析函数
            turn_progress = multi_agent_system.analyze_student_progress(dialogue_record)
            if turn_progress:
                # 将进度分析添加到对话记录中
                dialogue_record.progress_analysis = turn_progress
                
                # 输出调试信息
                if dialogue_record.debug_mode:
                    logger.info("\n📈 Student Progress Analysis:")
                    logger.info(f"Error Reduction: {turn_progress['error_reduction']}")
                    logger.info(f"Method Improvement: {turn_progress['method_improvement']}")
                    logger.info(f"Response Quality: {turn_progress['response_quality']}")
                    logger.info("Turn-by-turn progress:")
                    for turn_info in turn_progress['turn_progress']:
                        logger.info(f"  Turn {turn_info['turn']}:")
                        logger.info(f"    Method: {turn_info['method_used']}")
                        logger.info(f"    Errors: {', '.join(turn_info['errors']) if turn_info['errors'] else 'None'}")
                        logger.info(f"    Quality: {turn_info['response_quality']}")
                
                # 更新认知状态信息  # dddd
                # current_state = multi_agent_system.get_cognitive_state()
                # if initial_cognitive_state and current_state:
                #     logger.info(
                #         f"🧠 Cognitive State Updated - Carelessness: {initial_cognitive_state['carelessness_level']} -> {current_state['carelessness_level']}")

                # 记录本次对话的整体表现
                cognitive_progress.append({
                    'problem_index': total.item(),
                    'turns': dialogue_record.total_turns,
                    'success': dialogue_record.correct,
                    'progress_metrics': turn_progress,
                    'carelessness_trend': turn_progress.get('error_reduction', 'stable'),
                    'recent_success_rate': turn_progress.get('success_rate', 0.0)
                })
                
                logger.info(
                    f"📈 Progress: {turn_progress.get('error_reduction', 'stable')}, Success Rate: {turn_progress.get('success_rate', 0.0):.2f}")

        # 添加记录到实验记录器（包含进度分析）
        multi_agent_system.experiment_recorder.add_record(dialogue_record)

        # 计算一题多解分数（如果启用解题树）
        # 在对话结束后，输出解题树信息
        if config.agent.get("use_solution_tree", False) and multi_agent_system.current_solution_tree:
            solution_tree = multi_agent_system.current_solution_tree
            logger.info(f"🌳 解题树统计:")
            logger.info(f"   专家路径数: {len([p for p in solution_tree.solution_paths if p.get('type') == 'expert'])}")
            logger.info(f"   学生路径数: {len([p for p in solution_tree.solution_paths if p.get('type') == 'student'])}")

            if dialogue_record.debug_mode:
                logger.info("🌳 Visualizing Expert Tree:")
                vis_result = solution_tree.visualize_graph(f'expert_tree_{i}', owner='expert')
                logger.info(vis_result)

                logger.info("🌳 Visualizing Student Tree:")
                vis_result = solution_tree.visualize_graph(f'student_tree_{i}', owner='student')
                logger.info(vis_result)

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
    # multi_agent_system.visualize_cognitive_progress()  # dddd

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


def main(config, logger):
    val_dataset = create_raw_dataset(config.data.val_files, config.data)
    multi_agent_system = MultiAgentSystem(config)

    accuracy = solve_with_dialogue(config, logger, val_dataset, multi_agent_system)
    logger.info(f"\n🎯 Final Accuracy: {accuracy:.4f}")


if __name__ == '__main__':
    import hydra
    import os
    from omegaconf import OmegaConf
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    hydra.initialize(config_path="utils", version_base=None)

    overrides = []

    api_student = 'gpt-3.5-turbo-0125'  # GOOD imitator: gpt-3.5-turbo-0125   deepseek-v3
    api_teacher = 'deepseek-chat'  # deepseek-v3

    if 'deepseek' not in api_student or isinstance(ds_key_config.get(api_student, ""), str):
        os.environ["OPENAI_API_KEY"] = yi_key_config.get(api_student, "").get('key')
        os.environ["OPENAI_BASE_URL"] = yi_key_url
        api_student_url = 'https://api.apiyi.com/v1'
        api_student_key = yi_key_config.get(api_student, "").get('key')
    else:
        api_student_url = 'https://api.deepseek.com'
        api_student_key = ds_key_config.get(api_student, "").get('key')

    selected_model = 'deepseek-official'
    api_teacher_url = 'https://api.deepseek.com'
    api_teacher_key = ds_key_config.get(selected_model, "").get('key')
    os.environ["DEEPSEEK_API_KEY"] = ds_key_config.get(selected_model, "").get('key')
    os.environ["DEEPSEEK_BASE_URL"] = ds_key_url
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGSMITH_API_KEY"] = smith_key
    os.environ.pop('HTTP_PROXY', None)
    os.environ.pop('HTTPS_PROXY', None)
    os.environ.pop('http_proxy', None)
    os.environ.pop('https_proxy', None)
    os.environ.pop('ALL_PROXY', None)
    os.environ.pop('all_proxy', None)
    ''' 
    -----------------------------------
    ---------- 系统配置和超参数 ----------
    -----------------------------------
    '''
    overrides.append('data.val_files=D:\DeepLearning\Code\LangChain\dataset/APO_combine_with_source_test_without_path_3.parquet')
    # overrides.append('data.val_files=D:\DeepLearning\Code\LangChain\dataset\GSM8k_test_with_prompt4.parquet')
    # overrides.append('data.val_files=/mnt/t2-6tb/medical/SocraticLM_langchain/LangChain_3090/dataset/GSM8k_test_with_prompt4.parquet')
    overrides.append('+model.log_folder_path=D:\DeepLearning\Code\LangChain\log')
    # overrides.append('+model.log_folder_path=/mnt/t2-6tb/medical/SocraticLM_langchain/LangChain_3090/log')
    overrides.append('+agent.explicit_interaction=true')  # true=显式交互, false=工具调用
    overrides.append('data.prompt_key=extra_info')
    overrides.append('data.response_key=extra_info')
    overrides.append('+data.raw_problem_key=extra_info')
    overrides.append('+data.first_prompt=raw')
    overrides.append('+data.prompt_dict_keys=[question]')
    overrides.append('+data.response_dict_keys=[answer]')
    overrides.append('+data.raw_problem_dict_keys=[raw_problem]')
    overrides.append('+agent.max_samples=30')    # 测试用样本数量
    overrides.append('+agent.debug_samples=10')   # 查看完整response
    overrides.append('+agent.debug_mode=false')  # 查看完整prompt
    overrides.append('+agent.max_turns=5')

    ''' 
    -----------------------------------
    ---------- 智能体与对话配置 ----------
    -----------------------------------
    '''
    overrides.append(f'+model.api_name_student={api_student}')
    overrides.append('+model.model_type_student=api')
    overrides.append(f'+model.base_url_student={api_student_url}')
    overrides.append(f'+model.api_student_key={api_student_key}')
    # overrides.append('+model.model_type_student=local')
    # overrides.append('+model.api_name_student=qwen3-4b-4k:latest')  # qwen3-4b-4k:latest   qwen3:4b-tuned-4k
    # overrides.append('+model.base_url_student=http://localhost:11434')
    overrides.append('+model.temperature_student=0.7')
    overrides.append(f'+model.max_tokens_student={yi_key_config.get(api_student, "").get("max_token", 8000)}')  # 40000  8000

    overrides.append(f'+model.api_name_teacher={api_teacher}')
    overrides.append('+model.model_type_teacher=api')
    overrides.append(f'+model.base_url_teacher={api_teacher_url}')
    overrides.append(f'+model.api_teacher_key={api_teacher_key}')
    # overrides.append('+model.model_type_teacher=local')
    # overrides.append('+model.api_name_teacher=qwen3-4b-4k:latest')  # qwen3-4b-4k:latest   qwen3:4b-tuned-4k
    # overrides.append('+model.base_url_teacher=http://localhost:11434')
    overrides.append('+model.temperature_teacher=0.2')
    overrides.append(f'+model.max_tokens_teacher={ds_key_config.get(selected_model, "").get("max_token", 8000)}')  # 40000  8000

    ''' 
    -----------------------------------
    ------------ 解题树配置 -------------
    -----------------------------------
    '''
    overrides.append('+agent.use_solution_tree=true')                 # 启用解题树
    overrides.append('+agent.evaluate_multi_solution=false')           # 评估一题多解能力
    overrides.append(f'+model.api_name_expert={api_teacher}')
    overrides.append('+model.model_type_expert=api')
    overrides.append(f'+model.base_url_expert={api_teacher_url}')
    overrides.append(f'+model.api_key_expert={api_teacher_key}')
    # overrides.append('+model.model_type_expert=local')
    # overrides.append('+model.api_name_expert=qwen3-4b-4k:latest')  # qwen3-4b-4k:latest   qwen3:4b-tuned-4k
    overrides.append('+model.temperature_expert=0.2')
    overrides.append('+model.max_tokens_expert=8000')

    ''' 
    -----------------------------------
    ----------- 对话摘要配置 ------------
    -----------------------------------
    '''
    overrides.append('+agent.conversation_summary=false')  # 启用对话摘要
    overrides.append(f'+agent.summary_model_name={api_teacher}')
    overrides.append('+agent.summary_model_type=api')
    overrides.append(f'+agent.summary_base_url={api_teacher_url}')
    overrides.append(f'+agent.summary_api_key={api_teacher_key}')
    # overrides.append('+agent.summary_model_type=local')
    # overrides.append('+agent.summary_model_name=qwen3-4b-4k:latest')  # qwen3-4b-4k:latest   qwen3:4b-tuned-4k
    overrides.append('+model.temperature_summary_model=0')
    overrides.append('+model.max_tokens_summary_model=8000')
    overrides.append('+agent.summary_max_turns=4')  # 每6轮对话生成摘要
    overrides.append('+agent.summary_max_tokens=1500')  # 最大token限制
    overrides.append('+agent.summary_mode=per_message')  # or 'per_message'

    ''' 
    -----------------------------------
    ---------- Agent个性化配置 ----------
    -----------------------------------
    '''
    overrides.append('+agent.parallel_thinking=false')                 # 学生并行思考能力
    overrides.append('+agent.socratic_teaching=false')                 # 教师苏格拉底教学
    overrides.append('+agent.student_sys_prompt=STUDENT_PROMPT_EASY_MISTAKE')  # 选择学生系统提示词
    overrides.append('+agent.teacher_response=TEACHER_RESPONSE_TEACHING')  # 选择教师系统提示词
    overrides.append('+agent.student_response=STUDENT_RESPONSE_STUDYING_SOLVING')  # 选择学生系统提示词
    overrides.append('+agent.teacher_sys_prompt=TEACHER_PROMPT_EASY')
    overrides.append('+agent.teacher_tree_sys_prompt=TEACHER_PROMPT_EASY')  # 选择解题树系统提示词
    overrides.append('+agent.student_tree_sys_prompt=STUDENT_PROMPT_EASY')  # 选择解题树系统提示词
    overrides.append('+agent.expert_sys_prompt=EXPERT_STUDENT_PROMPT')  # 选择专家系统提示词
    
    overrides.append('+agent.use_cognitive_state=false')               # 学生认知状态配置 启用认知状态
    overrides.append('+agent.carelessness_level=9')                    # 粗心程度 (1-10)
    overrides.append('+agent.math_background=beginner')                    # 数学背景
    overrides.append('+agent.response_style=brief')               # 回复风格
    overrides.append('+agent.preferred_method=algebraic')              # 偏好方法
    overrides.append('+agent.learning_style=reading-writing')                   # 学习风格

    config = hydra.compose(config_name="sft_trainer", overrides=overrides)
    get_logger(config)
    logger.info(OmegaConf.to_yaml(config))
    main(config, logger)