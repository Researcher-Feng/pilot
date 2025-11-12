import re
import os

def extract_error_samples(log_file_path):
    """
    从日志文件中提取First Correct或Final Correct为False的样本编号
    
    Args:
        log_file_path (str): 日志文件路径
        
    Returns:
        list: 错误样本编号列表
    """
    error_samples = []
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"错误：找不到文件 {log_file_path}")
        return []
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return []
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # 匹配样本编号行，如：📊 Sample 1317/1319:
        sample_match = re.match(r'.*Sample\s+(\d+)/\d+:', line)
        
        if sample_match:
            sample_number = sample_match.group(1)
            
            # 检查接下来的几行中是否有First Correct或Final Correct为False
            first_correct_false = False
            final_correct_false = False
            
            # 检查接下来的5行（通常相关信息在样本编号行后面）
            for j in range(i + 1, min(i + 6, len(lines))):
                next_line = lines[j].strip()
                
                # 检查First Correct
                if 'First Correct: False' in next_line:
                    first_correct_false = True
                
                # 检查Final Correct
                if 'Final Correct: False' in next_line:
                    final_correct_false = True
            
            # 如果任一测试结果为False，则记录样本编号
            if first_correct_false or final_correct_false:
                error_samples.append(sample_number)
        
        i += 1
    
    return error_samples

def main():
    log_file_path = r"C:\Users\Pro14\Desktop\GSM8k__API_S_qwen3_4b-tuned-16k____API_T_deepseek-chat___log_20251106_102954.log"
    
    if not os.path.exists(log_file_path):
        print(f"文件不存在：{log_file_path}")
        return
    
    error_samples = extract_error_samples(log_file_path)
    
    if error_samples:
        print(f"找到 {len(error_samples)} 个错误样本：")
        for sample in error_samples:
            print(f"样本编号: {sample}")
        
        # 可选：将结果保存到文件
        output_file = r"C:\Users\Pro14\Desktop\error_samples.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in error_samples:
                    f.write(sample + '\n')
            print(f"\n错误样本编号已保存到：{output_file}")
        except Exception as e:
            print(f"保存结果文件时出错：{e}")
    else:
        print("未找到错误样本")


import pandas as pd
def create_error_samples_file(log_file_path, pd_file_path, new_pd_file_path):
    with open(log_file_path, 'r', encoding='utf-8') as file:
        error_lines = [line.strip() for line in file.readlines()]
    dataframe = pd.read_parquet(pd_file_path)
    
    error_indices = []
    for index, row in dataframe.iterrows():
        if str(index) in error_lines:
            error_indices.append(index)
    
    error_dataframe = dataframe.loc[error_indices]
    error_dataframe.to_parquet(new_pd_file_path)


def check_samples(pd_file_path):
    dataframe = pd.read_parquet(pd_file_path)
    print(len(dataframe))
    for index, row in dataframe.iterrows():
        if 'APO_AIME25' != row['data_source']:
            print(index)
        if 'APO_AMC23' != row['data_source']:
            print(index)
        if 'APO_AIME24' != row['data_source']:
            print(index)
        pass


if __name__ == "__main__":
    # main()
    log_file_path = r"C:\Users\Pro14\Desktop\error_samples.txt"
    pd_file_path = r"D:\DeepLearning\Code\LangChain\dataset/math_test_with_prompt_level4_4.parquet"
    new_pd_file_path = r"D:\DeepLearning\Code\LangChain\dataset/math_test_with_prompt_level4_error.parquet"
    # create_error_samples_file(log_file_path, pd_file_path, new_pd_file_path)
    # check_pd_file = rf"D:\DeepLearning\Code\LangChain\dataset\APO_combine_with_source_test_without_path_3.parquet"
    check_pd_file = rf"D:\DeepLearning\Code\LangChain\dataset\GSM8k_test_with_prompt4.parquet"
    check_samples(check_pd_file)
    