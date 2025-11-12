import sys
from datetime import datetime
import re
import os
import logging

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "INFO"))
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_logger(config):
    global logger
    eval_have_api_student = f'_API_S_{config.model.api_name_student}__' if config.model.get("api_name_student", None) else '_API_N__'
    eval_have_api_teacher = f'_API_T_{config.model.api_name_teacher}__' if config.model.get("api_name_teacher", None) else '_API_N__'
    if 'gsm' in config.data.val_files.lower():
        eval_data = 'GSM8k'
    elif 'math' in config.data.val_files.lower():
        eval_data = 'MATH'
    elif 'apo' in config.data.val_files.lower():
        eval_data = 'APO'
    else:
        eval_data = 'UNKNOWN'

    log_filename = f"{eval_data}_{eval_have_api_student}_{eval_have_api_teacher}_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"  # 示例文件名
    log_filename = log_filename.replace(':', '_')
    log_path = os.path.join(config.model.log_folder_path, log_filename)
    os.makedirs(config.model.log_folder_path, exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')  # 使用追加模式
    file_handler.setLevel(logging.INFO)  # 设置文件处理器的日志级别
    file_handler.setFormatter(formatter)  # 为文件处理器设置格式
    console_handler = logging.StreamHandler(sys.stdout)
    # 确保logger没有重复的处理器（避免日志重复输出）
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)


def extract_boxed(output):
    lidx = output.rfind("\\boxed{")
    if lidx == -1:
        return None
    lidx += len("\\boxed{")
    ridx = output.find("}", lidx)
    return_ans = output[lidx:ridx]
    if return_ans.count("{") == return_ans.count("}"):
        return return_ans
    ridx = output.rfind("}", lidx)
    return output[lidx:ridx] if ridx != -1 else None


def extract_answer(text):
    """
    按优先级提取答案
    """
    reg_num = r'\s*(\d+(?:/\d+)?)(?:,\d+)*'
    reg_any = r'\s*([^\s]+)'
    reg_many = r'\s*((?:[-+]?\d*[.,]?\d+[\s\S]*?)|(?:\$[^$]+\$)|(?:\\\([\s\S]*?\\\))|(?:\\\[[\s\S]*?\\\]))(?=,|$|\s*[a-zA-Z]+\s*=)'
    # 方法1: 提取 #### 后面的数字
    well_match = re.search(rf'####{reg_num}', text)
    ref_number = well_match.group(1) if well_match else None
    if ref_number:
        return str(ref_number)

    # 方法2: 提取 \\boxed{} 中的内容
    box_match = extract_boxed(text)
    if box_match:
        return str(box_match)

    # 方法3: 提取 Final Answer: 后面的内容
    boxed_match2 = re.search(rf'Final Answer\s*:{reg_any}', text)
    ref_number2 = boxed_match2.group(1) if boxed_match2 else None
    if ref_number2:
        return str(ref_number2)

    # 方法4: 如果以上方法都找不到，查找文本中的最后一个数字
    all_equals = re.findall(reg_num, text)

    if all_equals:
        last_expression = all_equals[-1].strip()
        return str(last_expression)

    return None



if __name__ == "__main__":
    t = rf"Janet\ makes\ $18\ every\ day\ at\ the\ farmers'\ market."
    print(extract_answer(t))


