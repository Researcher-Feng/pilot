from function.agent_IO import SimpleAgent
from prompt.system import *
from function.tools_fun import *
from function.contex_fun import *
from function.format_fun import *
from function.memory_fun import *
from utils.evaluator import *
# from utils.dataset.tokenizer import hf_tokenizer
# from utils.dataset.fs import copy_to_local
from utils.dataset.parallel_thinking_sft_dataset import *
from utils.MARIO_EVAL.demo import is_equiv_MATH

from langchain.agents.structured_output import ToolStrategy, ProviderStrategy
from tqdm import tqdm
import torch
device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device_name}")


def create_raw_dataset(data_paths, data_config):
    dataset = RawDataset(parquet_file=data_paths, config=data_config)
    return dataset


def solve_with_dialogue(config, val_dataset, agent_student, agent_teacher):
    # logger = get_logger(config)

    eval_have_api_student = f'API: {config.model.api_name_student}' if config.model.get("api_name_student", None) else 'API: N'
    eval_have_api_teacher = f'API: {config.model.api_name_teacher}' if config.model.get("api_name_teacher", None) else 'API: N'
    eval_data = 'GSM8k' if 'gsm' in config.data.val_files.lower() else 'MATH'
    desc = f"Evaluating Accuracy of config [{eval_have_api_student}, {eval_have_api_teacher}, {eval_data}]"
    correct = torch.zeros(1, dtype=torch.float32, device=device_name)
    total = torch.zeros(1, dtype=torch.float32, device=device_name)
    error_ans = torch.zeros(1, dtype=torch.float32, device=device_name)

    for i, prompt in tqdm(enumerate(val_dataset.prompts), desc=desc):
        label_ans = val_dataset.responses[i]
        agent_student.chat_once(prompt, silence=True)
        gt = agent_student.response['messages'][-1].content
        # agent_teacher.chat_once(prompt)

        total += 1
        gen_ans = extract_answer(gt)
        true_ans = extract_answer(label_ans)
        if not gen_ans or not true_ans:
            error_ans += 1
            continue
        if is_equiv_MATH(true_ans, gen_ans):
            correct += 1
        else:
            logger.info(f'prompts (sample): {prompt}\n\n')
            logger.info(f'generated_texts (sample): {gt}\n\n')
            logger.info(f'label_ans (sample): {label_ans}')
            logger.info(f'gen_ans (sample): {gen_ans}')
            logger.info(f'true_ans (sample): {true_ans}')

        logger.info(
            f'correct: {correct.item() / total.item():.4f}\terror_ans: {error_ans.item()}')


def get_agent(prompt_sys, api_name):
    assert prompt_sys is not None
    agent = SimpleAgent()
    agent.model_init(model_name=api_name, model_type='basic')
    agent.config_create("thread_id", "1")
    '''
    Use middleware to define custom state when your custom state needs to be accessed
    by specific middleware hooks and tools attached to said middleware.
    '''
    agent.agent_init(prompt_sys_name=prompt_sys)
    # user1 = r'I prefer technical explanations. What is MATH?'
    agent.context_set(user_id="1")
    # agent.chat_once(user1)
    return agent


def main(config):
    val_dataset = create_raw_dataset(config.data.val_files, config.data)
    if config.model.api_name_student and config.model.api_name_teacher:
        agent_student = get_agent(STUDENT_PROMPT, config.model.api_name_student)
        agent_teacher = get_agent(TEACHER_PROMPT, config.model.api_name_teacher)
    else:
        raise NotImplemented
    solve_with_dialogue(config, val_dataset, agent_student, agent_teacher)


if __name__ == '__main__':
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    # Initialize Hydra and compose config programmatically
    GlobalHydra.instance().clear()  # Clear any previous init
    hydra.initialize(config_path="utils", version_base=None)  # Point to your config dir

    # Compose with base config and overrides (list of strings mimicking .sh args)
    overrides = [
        'data.val_files=D:\DeepLearning\Code\LangChain\dataset/GSM8k_test_with_prompt2.parquet',
        'data.prompt_key=extra_info',
        'data.response_key=extra_info',
        'data.max_length=4096',
        '+data.first_prompt=raw',
        '+data.prompt_dict_keys=[question]',
        '+data.response_dict_keys=[answer]',
        '+model.api_name_student=deepseek-chat',
        '+model.api_name_teacher=deepseek-chat',
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



