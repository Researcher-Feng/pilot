# run.py
from function.agent_IO import SimpleAgent, ModelConfig
from prompt.system import *
from function.tools_fun import *
from function.contex_fun import *
from function.format_fun import *
from function.memory_fun import *
from utils.evaluator import *
from utils.MARIO_EVAL.demo import is_equiv_MATH
from utils.dataset.parallel_thinking_sft_dataset import RawDataset

from tqdm import tqdm
import torch
device_name = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device_name}")


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
            base_url=self.config.model.get("base_url_student", "http://localhost:11434"),
            temperature=self.config.model.get("temperature_student", 0.7),
            max_tokens=self.config.model.get("max_tokens_student", 2000)
        )

        # 教师模型配置
        teacher_model_config = ModelConfig(
            model_type=self.config.model.get("model_type_teacher", "api"),
            model_name=self.config.model.api_name_teacher,
            base_url=self.config.model.get("base_url_teacher", "http://localhost:11434"),
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


def create_raw_dataset(data_paths, data_config):
    # 保持原有实现
    dataset = RawDataset(parquet_file=data_paths, config=data_config)
    return dataset


def solve_with_dialogue(config, logger, val_dataset, multi_agent_system):
    """使用多智能体对话解决问题"""
    correct = torch.zeros(1, dtype=torch.float32, device=device_name)
    total = torch.zeros(1, dtype=torch.float32, device=device_name)
    error_ans = torch.zeros(1, dtype=torch.float32, device=device_name)

    student_agent, teacher_agent = multi_agent_system.initialize_agents()

    explicit_mode = config.agent.get("explicit_interaction", True)
    eval_mode = 'Explicit Interaction' if explicit_mode else 'Tool-based'

    desc = f"Multi-agent Evaluation [{eval_mode}] - Dataset: {'GSM8k' if 'gsm' in config.data.val_files.lower() else 'MATH'}"

    for i, prompt in tqdm(enumerate(val_dataset.prompts), desc=desc):
        if i >= config.agent.get("max_samples", 10):  # 限制样本数量用于测试
            break

        label_ans = val_dataset.responses[i]

        if explicit_mode:
            # 模式1: 显式交互
            final_answer = student_agent.multi_agent_chat_explicit(teacher_agent, prompt)
        else:
            # 模式2: 工具调用
            final_answer = student_agent.multi_agent_chat_tool_based(prompt)

        total += 1
        gen_ans = extract_answer(final_answer)
        true_ans = extract_answer(label_ans)

        if not gen_ans or not true_ans:
            error_ans += 1
            continue

        if is_equiv_MATH(true_ans, gen_ans):
            correct += 1
            logger.info(f"✅ Correct answer for sample {i}")
        else:
            logger.info(f"❌ Incorrect answer for sample {i}")
            logger.info(f'   Problem: {prompt[:100]}...')
            logger.info(f'   Generated: {final_answer[:100]}...')
            logger.info(f'   Expected: {label_ans[:100]}...')

        current_accuracy = correct.item() / total.item() if total.item() > 0 else 0
        logger.info(f'   Current Accuracy: {current_accuracy:.4f}')

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
        'data.val_files=D:\DeepLearning\Code\LangChain\dataset/GSM8k_test_with_prompt2.parquet',
        # 'data.val_files=/mnt/t2-6tb/medical/SocraticLM_langchain/LangChain_3090/dataset/GSM8k_test_with_prompt2.parquet',
        'data.prompt_key=extra_info',
        'data.response_key=extra_info',
        'data.max_length=4096',
        '+data.first_prompt=raw',
        '+data.prompt_dict_keys=[question]',
        '+data.response_dict_keys=[answer]',

        # 模型配置
        # '+model.api_name_student=deepseek-chat',
        # '+model.model_type_student=api',
        '+model.model_type_student=local',
        '+model.api_name_student=qwen3-4b-4k:latest',
        '+model.base_url_student=http://localhost:11434',

        '+model.api_name_teacher=deepseek-chat',
        '+model.model_type_teacher=api',
        # '+model.model_type_teacher=local',
        # '+model.api_name_teacher=qwen2.5:0.5b',
        # '+model.base_url_teacher=http://localhost:11434',

        # 智能体配置 - 功能开关
        '+agent.max_turns=5',
        '+agent.max_samples=10',  # 测试用样本数量
        '+agent.explicit_interaction=true',  # true=显式交互, false=工具调用
        '+agent.parallel_thinking=false',  # 学生并行思考能力
        '+agent.socratic_teaching=true',  # 教师苏格拉底教学
        '+agent.math_background=intermediate',  # beginner/intermediate/advanced

        # 'model.partial_pretrain=/mnt/t2-6tb/medical/pretrained/Qwen3_merged_with_lora_global_step_14775',
        '+model.log_folder_path=D:\DeepLearning\Code\LangChain\log',
        # '+model.log_folder_path=/mnt/t2-6tb/medical/SocraticLM_langchain/LangChain_3090/log',
        '+trainer.checkpoint_path=True',
        'trainer.logger=[console]',
        'generate.max_new_tokens=1024',
        '+generate.do_sample=false',
        '+generate.num_beams=1',
        '+generate.temperature=1.0',
        '+generate.top_p=1.0',
    ]

    config = hydra.compose(config_name="sft_trainer", overrides=overrides)
    logger = get_logger(config)
    logger.info(OmegaConf.to_yaml(config))
    main(config, logger)