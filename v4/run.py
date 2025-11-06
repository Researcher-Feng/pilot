# run.py
from function.agent_IO import SimpleAgent, ModelConfig, DialogueRecord, ExperimentRecorder
from prompt.system import *
from function.tools_fun import *
from function.contex_fun import *
from function.format_fun import *
from function.memory_fun import *
from utils.evaluator import *
from utils.MARIO_EVAL.demo import is_equiv_MATH
from utils.dataset.parallel_thinking_sft_dataset import RawDataset

from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import seaborn as sns
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
        exp_name = f"math_tutoring_explicit_interaction" if config.agent.explicit_interaction else "math_tutoring_explicit"
        self.experiment_recorder = ExperimentRecorder(exp_name)

    def initialize_agents(self):
        """初始化学生和教师智能体"""
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

        # 初始化学生智能体
        self.student_agent = SimpleAgent(agent_type="student")
        self.student_agent.model_init(student_model_config)
        self.student_agent.config_create("thread_id", "student_1")

        # 初始化教师智能体
        self.teacher_agent = SimpleAgent(agent_type="teacher")
        self.teacher_agent.model_init(teacher_model_config)
        self.teacher_agent.config_create("thread_id", "teacher_1")

        # 配置智能体参数
        student_kwargs = {
            "max_turns": self.config.agent.get("max_turns", 5),
            "parallel_thinking": self.config.agent.get("parallel_thinking", False),
            "math_background": self.config.agent.get("math_background", "intermediate")
        }

        teacher_kwargs = {
            "socratic_teaching": self.config.agent.get("socratic_teaching", True)
        }

        # 工具配置
        student_tools = []
        teacher_tools = []

        if self.config.agent.get("parallel_thinking", False):
            student_tools.append(parallel_thinking)

        if self.config.agent.get("socratic_teaching", True):
            teacher_tools.append(socratic_questioning)
            teacher_tools.append(math_concept_explainer)

        self.student_agent.agent_init(
            student_model_config,
            prompt_sys_name=STUDENT_PROMPT,
            tools_list=student_tools,
            **student_kwargs
        )
        self.teacher_agent.agent_init(
            teacher_model_config,
            prompt_sys_name=TEACHER_PROMPT,
            tools_list=teacher_tools,
            **teacher_kwargs
        )

        # 设置上下文
        self.student_agent.context_set(
            user_id="student_1",
            user_role="student",
            math_background=student_kwargs["math_background"],
            parallel_thinking=student_kwargs["parallel_thinking"],
            conversation_mode="explicit" if self.config.agent.get("explicit_interaction", True) else "tool_based"
        )
        self.teacher_agent.context_set(
            user_id="teacher_1",
            user_role="teacher",
            socratic_teaching=teacher_kwargs["socratic_teaching"]
        )

        logger.info("✅ Multi-agent system initialized successfully!")
        logger.info(f"   Student: {self.config.model.api_name_student}")
        logger.info(f"   Teacher: {self.config.model.api_name_teacher}")
        logger.info(
            f"   Mode: {'Explicit Interaction' if self.config.agent.get('explicit_interaction', True) else 'Tool-based'}")
        logger.info(f"   Parallel Thinking: {student_kwargs['parallel_thinking']}")
        logger.info(f"   Socratic Teaching: {teacher_kwargs['socratic_teaching']}")
        logger.info(f"   Math Background: {student_kwargs['math_background']}")

        return self.student_agent, self.teacher_agent

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
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
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


def create_raw_dataset(data_paths, data_config):
    # 保持原有实现
    dataset = RawDataset(parquet_file=data_paths, config=data_config)
    return dataset


def solve_with_dialogue(config, logger, val_dataset, multi_agent_system):
    """使用多智能体对话解决问题"""
    first_correct = torch.zeros(1, dtype=torch.float32, device=device_name)
    correct = torch.zeros(1, dtype=torch.float32, device=device_name)
    total = torch.zeros(1, dtype=torch.float32, device=device_name)
    error_ans = torch.zeros(1, dtype=torch.float32, device=device_name)

    student_agent, teacher_agent = multi_agent_system.initialize_agents()

    explicit_mode = config.agent.get("explicit_interaction", True)
    eval_mode = 'Explicit Interaction' if explicit_mode else 'Tool-based'

    max_samples = config.agent.get("max_samples", min(10, len(val_dataset.prompts)))

    desc = f"Multi-agent Evaluation [{eval_mode}] - Dataset: {'GSM8k' if 'gsm' in config.data.val_files.lower() else 'MATH'}"

    for i, (prompt, label_ans) in tqdm(
            enumerate(zip(val_dataset.prompts[:max_samples], val_dataset.responses[:max_samples])),
            desc=desc, total=max_samples
    ):
        # 创建对话记录
        dialogue_record = DialogueRecord(prompt, label_ans, debug_mode=True if i < config.agent.debug_samples else False)

        if explicit_mode:
            # 模式1: 显式交互
            final_answer, correct_answer, dialogue_record = student_agent.multi_agent_chat_explicit(
                teacher_agent, prompt, label_ans, dialogue_record
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
        else:
            if dialogue_record.first_correct:
                first_correct += 1
            if dialogue_record.correct:
                correct += 1

        # 添加记录到实验记录器
        multi_agent_system.experiment_recorder.add_record(dialogue_record)

        # 打印当前进度
        current_first_accuracy = first_correct.item() / total.item() if total.item() > 0 else 0
        current_accuracy = correct.item() / total.item() if total.item() > 0 else 0
        logger.info(f'\n📊 Sample {i + 1}/{max_samples}:')
        logger.info(f'   First Correct: {dialogue_record.first_correct}')
        logger.info(f'   Final Correct: {dialogue_record.correct}')
        logger.info(f'   Total Dialogue Round: {dialogue_record.total_turns}')
        logger.info(f'   Parallel Thinking Count: {dialogue_record.parallel_thinking_count}')
        logger.info(f'   Thinking Paths Count: {dialogue_record.thinking_paths_count}')
        logger.info(f'   Leaked Answer: {dialogue_record.leaked_answer}')
        logger.info(f'   Current First Accuracy: {current_first_accuracy:.4f}')
        logger.info(f'   Current Final Accuracy: {current_accuracy:.4f}')

    # 计算最终统计并保存结果
    multi_agent_system.experiment_recorder.calculate_statistics()
    multi_agent_system.experiment_recorder.save_results()
    multi_agent_system.experiment_recorder.print_summary()

    # 生成可视化
    multi_agent_system.visualize_results()

    final_accuracy = correct.item() / total.item() if total.item() > 0 else 0
    logger.info(f"\n🎯 Final Results:")
    logger.info(f"   Total Samples: {total.item()}")
    logger.info(f"   Correct Answers: {correct.item()}")
    logger.info(f"   Error Answers: {error_ans.item()}")
    logger.info(f"   Final Accuracy: {final_accuracy:.4f}")

    return final_accuracy


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

    overrides = [
        # 'data.val_files=D:\DeepLearning\Code\LangChain\dataset/GSM8k_test_with_prompt2.parquet',
        'data.val_files=/mnt/t2-6tb/medical/SocraticLM_langchain/LangChain_3090/dataset/GSM8k_test_with_prompt2.parquet',
        'data.prompt_key=extra_info',
        'data.response_key=extra_info',
        'data.max_length=4096',
        '+data.first_prompt=raw',
        '+data.prompt_dict_keys=[question]',
        '+data.response_dict_keys=[answer]',

        # 模型配置
        '+model.temperature_student=0.7',
        '+model.max_tokens_student=2000',
        '+model.temperature_teacher=0.7',
        '+model.max_tokens_teacher=2000',

        '+model.api_name_student=deepseek-chat',
        '+model.model_type_student=api',
        # '+model.model_type_student=local',
        # '+model.api_name_student=qwen3:4b-tuned-4k',  # qwen3-4b-4k:latest
        # '+model.base_url_student=http://localhost:11434',

        '+model.api_name_teacher=deepseek-chat',
        '+model.model_type_teacher=api',
        # '+model.model_type_teacher=local',
        # '+model.api_name_teacher=qwen2.5:0.5b',
        # '+model.base_url_teacher=http://localhost:11434',

        # 智能体配置 - 功能开关
        '+agent.max_turns=3',
        '+agent.max_samples=10',  # 测试用样本数量
        '+agent.debug_samples=2',
        '+agent.explicit_interaction=true',  # true=显式交互, false=工具调用
        '+agent.parallel_thinking=false',  # 学生并行思考能力
        '+agent.socratic_teaching=false',  # 教师苏格拉底教学
        '+agent.math_background=intermediate',  # beginner/intermediate/advanced

        # 'model.partial_pretrain=/mnt/t2-6tb/medical/pretrained/Qwen3_merged_with_lora_global_step_14775',
        # '+model.log_folder_path=D:\DeepLearning\Code\LangChain\log',
        '+model.log_folder_path=/mnt/t2-6tb/medical/SocraticLM_langchain/LangChain_3090/log',
        '+trainer.checkpoint_path=True',
    ]

    config = hydra.compose(config_name="sft_trainer", overrides=overrides)
    get_logger(config)
    logger.info(OmegaConf.to_yaml(config))
    main(config, logger)