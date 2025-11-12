import pandas as pd


def process_gsm8k_data(input_file, output_file):
    """处理GSM8k数据文件，修改两个字段的内容并保存为新文件"""

    def move_final_answer_to_beginning(text):
        """将结尾的提示语移动到开头 - 用于question列"""
        solve_sentence = "Solve the following problem step by step."
        target_sentence = "Put your final answer within the \\boxed{} command."
        if text.endswith(target_sentence):
            # 移除结尾的提示语
            main_content = text[:-len(target_sentence)].strip()
            # 将提示语放到开头
            return f"{solve_sentence} {target_sentence} \n\n {main_content}"
        else:
            return f"{solve_sentence} {target_sentence} \n\n {text}"


    def move_final_answer_after_solve(text):
        """将结尾的提示语移动到'Solve the following problem step by step.'后面 - 用于prompt字段"""
        solve_sentence = "Solve the following problem step by step."
        target_sentence = "Put your final answer within the \\boxed{} command."

        if text.endswith(target_sentence) and solve_sentence in text:
            # 移除结尾的提示语
            remaining_content = text[:-len(target_sentence)].strip()
            # 找到solve句子的位置
            solve_index = remaining_content.find(solve_sentence)
            if solve_index != -1:
                # 在solve句子后插入提示语
                before_solve = remaining_content[:solve_index + len(solve_sentence)]
                after_solve = remaining_content[solve_index + len(solve_sentence):].strip()
                return f"{before_solve} {target_sentence} {after_solve}"
        return text

    def get_raw_problem(text):
        """将结尾的提示语移动到'Solve the following problem step by step.'后面 - 用于prompt字段"""
        solve_sentence = "Solve the following problem step by step."
        target_sentence = "Put your final answer within the \\boxed{} command."

        if text.endswith(target_sentence) and solve_sentence in text:
            # 移除结尾的提示语
            remaining_content = text[:-len(target_sentence)].strip()
            raw_problem = remaining_content[remaining_content.find('\nProblem: ') + len('\nProblem: '):]
            return raw_problem
        return text

    print(f"正在读取文件: {input_file}")
    dataframe = pd.read_parquet(input_file)

    print("开始处理数据...")

    # 处理第一个字段：dataframe['question']列
    if 'question' in dataframe.columns:
        print("处理 question 列...")
        dataframe['question'] = dataframe['question'].apply(move_final_answer_to_beginning)

    # 处理第二个字段：prompt_key字段中的question
    # 假设prompt_key是数据框中的一个列，包含字典结构
    prompt_key = 'extra_info'  # 根据实际情况调整这个键名
    if prompt_key in dataframe.columns:
        print(f"处理 {prompt_key} 字段中的question...")

        def process_prompt_item(x):
            """处理prompt字段中的嵌套结构"""
            if isinstance(x, dict) and 'question' in x:
                # 处理嵌套在extra_info中的question
                original_question = x['question']
                processed_question = move_final_answer_after_solve(original_question)
                processed_raw_problem = get_raw_problem(original_question)
                # 创建新的字典副本，避免修改原始数据
                new_x = x.copy()
                new_x['question'] = processed_question
                new_x['raw_problem'] = processed_raw_problem
                return new_x
            elif isinstance(x, str):
                # 如果是字符串，直接处理
                return move_final_answer_after_solve(x)
            return x

        dataframe[prompt_key] = dataframe[prompt_key].apply(process_prompt_item)

    # 保存处理后的数据
    print(f"保存处理后的数据到: {output_file}")  # dataframe[prompt_key].iloc[0]['question'], dataframe['question'][0]
    dataframe.to_parquet(output_file, index=False)

    print("数据处理完成！")

    # 验证处理结果
    print("\n验证处理结果:")
    print(f"原始数据形状: {dataframe.shape}")
    print("前3行数据预览:")
    print(dataframe.head(3))

    return dataframe


# def process_apo_data(input_file, output_file):
#     dataframe = pd.read_parquet(input_file)
#     for index, row in dataframe.iterrows():
#         # for each row, move row['prompt']['content'] to row['question']
#         dataframe.at[index, 'question'] = row['prompt']['content']
#         # for each row, move row['reward_model']['ground_truth'] to row['extra_info']['answer']
#         dataframe.at[index, 'extra_info']['answer'] = row['reward_model']['ground_truth']
#     dataframe.to_parquet(output_file, index=False)
#     return dataframe

def process_apo_data(input_file, output_file):
    dataframe = pd.read_parquet(input_file)
    for index, row in dataframe.iterrows():
        # for each row, move row['prompt'][0]['content'] to row['question']
        # (prompt is a list, so access the first item)
        dataframe.at[index, 'question'] = row['prompt'][0]['content']
        
        # for each row, move row['reward_model']['ground_truth'] to row['extra_info']['answer']
        # Need to create a copy of the dict and modify it
        extra_info_copy = row['extra_info'].copy()
        extra_info_copy['answer'] = row['reward_model']['ground_truth']
        extra_info_copy['raw_problem'] = row['prompt'][0]['content']
        dataframe.at[index, 'extra_info'] = extra_info_copy
    
    dataframe.to_parquet(output_file, index=False)
    return dataframe


# 执行处理函数
if __name__ == "__main__":
    input_file = r'D:\DeepLearning\Code\LangChain\dataset/APO_combine_with_source_test_without_path_2.parquet'
    output_file = r'D:\DeepLearning\Code\LangChain\dataset/APO_combine_with_source_test_without_path_3.parquet'
    processed_data = process_gsm8k_data(input_file, output_file)
    # process_apo_data(input_file, output_file)