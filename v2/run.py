# run.py
from v2.function.agent_IO import SimpleAgent, ModelConfig
from v2.prompt.system import *
from v2.utils import *
from v2.utils.MARIO_EVAL.demo import is_equiv_MATH
from v2.utils.dataset.parallel_thinking_sft_dataset import RawDataset

from tqdm import tqdm
import torch
device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_name}")


class MultiAgentSystem:
    """多智能体系统管理器"""

    def __init__(self, config):
        self.config = config
        self.student_agent = None
        self.teacher_agent = None

    def initialize_agents(self):
        """初始化学生和教师智能体"""
        # 学生模型配置
        student_model_config = ModelConfig(
            model_type=self.config.model.get("model_type_student", "api"),
            model_name=self.config.model.api_name_student,
            base_url=self.config.model.get("base_url_student", ""),
            temperature=self.config.model.get("temperature_student", 0.7),
            max_tokens=self.config.model.get("max_tokens_student", 2000)
        )

        # 教师模型配置
        teacher_model_config = ModelConfig(
            model_type=self.config.model.get("model_type_teacher", "api"),
            model_name=self.config.model.api_name_teacher,
            base_url=self.config.model.get("base_url_teacher", ""),
            temperature=self.config.model.get("temperature_teacher", 0.3),
            max_tokens=self.config.model.get("max_tokens_teacher", 2000)
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

        self.student_agent.agent_init(
            prompt_sys_name=STUDENT_PROMPT,
            **student_kwargs
        )
        self.teacher_agent.agent_init(
            prompt_sys_name=TEACHER_PROMPT,
            **teacher_kwargs
        )

        self.student_agent.context_set(
            user_id="student_1",
            user_role="student",
            math_background=student_kwargs["math_background"]
        )
        self.teacher_agent.context_set(
            user_id="teacher_1",
            user_role="teacher"
        )

        return self.student_agent, self.teacher_agent


def create_raw_dataset(data_paths, data_config):
    # 保持原有实现
    dataset = RawDataset(parquet_file=data_paths, config=data_config)
    return dataset


def solve_with_dialogue(config, val_dataset, multi_agent_system):
    """使用多智能体对话解决问题"""
    correct = torch.zeros(1, dtype=torch.float32, device=device_name)
    total = torch.zeros(1, dtype=torch.float32, device=device_name)
    error_ans = torch.zeros(1, dtype=torch.float32, device=device_name)

    student_agent, teacher_agent = multi_agent_system.initialize_agents()

    eval_mode = '显式交互' if config.agent.get("explicit_interaction", True) else '工具调用'
    desc = f"多智能体评估 [{eval_mode}] - 数据集: {'GSM8k' if 'gsm' in config.data.val_files.lower() else 'MATH'}"

    for i, prompt in tqdm(enumerate(val_dataset.prompts), desc=desc):
        label_ans = val_dataset.responses[i]

        if config.agent.get("explicit_interaction", True):
            # 模式1: 显式交互
            final_answer = student_agent.multi_agent_chat(teacher_agent, prompt)
        else:
            # 模式2: 工具调用 (将在下一步实现)
            final_answer = student_agent.chat_once(prompt)

        total += 1
        gen_ans = extract_answer(final_answer)
        true_ans = extract_answer(label_ans)

        if not gen_ans or not true_ans:
            error_ans += 1
            continue

        if is_equiv_MATH(true_ans, gen_ans):
            correct += 1
        else:
            print(f'prompts (sample): {prompt}')
            print(f'generated_texts (sample): {final_answer}')
            print(f'label_ans (sample): {label_ans}')
            print(f'gen_ans (sample): {gen_ans}')
            print(f'true_ans (sample): {true_ans}')

        print(f'准确率: {correct.item() / total.item():.4f}\t错误答案: {error_ans.item()}')

    return correct.item() / total.item() if total.item() > 0 else 0


def main(config):
    val_dataset = create_raw_dataset(config.data.val_files, config.data)
    multi_agent_system = MultiAgentSystem(config)

    accuracy = solve_with_dialogue(config, val_dataset, multi_agent_system)
    print(f"\n🎯 最终准确率: {accuracy:.4f}")


if __name__ == '__main__':
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    hydra.initialize(config_path="../v1/utils", version_base=None)

    overrides = [
        'data.val_files=D:\DeepLearning\Code\LangChain\dataset/GSM8k_test_with_prompt2.parquet',
        'data.prompt_key=extra_info',
        'data.response_key=extra_info',
        'data.max_length=4096',
        '+data.first_prompt=raw',
        '+data.prompt_dict_keys=[question]',
        '+data.response_dict_keys=[answer]',

        # 模型配置
        '+model.api_name_student=deepseek-chat',
        '+model.api_name_teacher=deepseek-chat',
        '+model.model_type_student=api',
        '+model.model_type_teacher=api',

        # 本地模型示例配置 (取消注释以使用)
        # '+model.model_type_student=local',
        # '+model.api_name_student=qwen:7b',
        # '+model.base_url_student=http://localhost:11434',

        # 智能体配置
        '+agent.max_turns=5',
        '+agent.explicit_interaction=true',
        '+agent.parallel_thinking=false',
        '+agent.socratic_teaching=true',
        '+agent.math_background=intermediate',

        'model.partial_pretrain=/mnt/t2-6tb/medical/pretrained/Qwen3_merged_with_lora_global_step_14775',
        '+model.log_folder_path=D:\DeepLearning\Code\LangChain\log',
        '+trainer.checkpoint_path=True',
        'trainer.logger=[console]',
        'generate.max_new_tokens=1024',
        '+generate.do_sample=false',
        '+generate.num_beams=1',
        '+generate.temperature=1.0',
        '+generate.top_p=1.0',
    ]

    config = hydra.compose(config_name="sft_trainer", overrides=overrides)
    main(config)