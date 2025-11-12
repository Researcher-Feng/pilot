import datetime
from typing import Dict, Any


class DialogueRecord:
    """对话记录类"""

    def __init__(self, problem: str, correct_answer: str, debug_mode: bool = False):
        self.problem = problem
        self.correct_answer = correct_answer
        self.debug_mode = debug_mode
        self.turns = []
        self.final_student_answer = ""
        self.first_correct = False
        self.correct = False
        self.leaked_answer = False
        self.parallel_thinking_count = 0
        self.thinking_paths_count = 0
        self.total_turns = 0
        self.progress_analysis = None  # 存储对话内进步分析

    def add_turn(self, turn_data: Dict[str, Any]):
        """添加一轮对话记录"""
        self.turns.append(turn_data)
        self.total_turns = len(self.turns)

    def analyze_student_response(self, response: str):
        """分析学生回复"""
        # 统计并行思考标签
        parallel_count = response.count('<Parallel')
        self.parallel_thinking_count += parallel_count

        # 统计思考路径标签
        path_count = response.count('<Path')
        self.thinking_paths_count += path_count

        return parallel_count, path_count

    def check_answer_leakage(self, teacher_response: str, answer_num: str):
        """检查教师是否泄露答案"""
        # 简单的答案泄露检测逻辑
        leakage_indicators = [
            "Final Answer: " + self.correct_answer,
            "the result is " + self.correct_answer,
            "equals to " + self.correct_answer,
            "= " + self.correct_answer,
            answer_num
        ]

        leakage_detected = any(
            indicator.lower() in teacher_response.lower()
            for indicator in leakage_indicators
            if indicator.strip()
        )

        if leakage_detected:
            self.leaked_answer = True
        else:
            self.leaked_answer = False

        return leakage_detected

    def to_dict(self):
        """转换为字典格式"""
        data = {
            "problem": self.problem,
            "correct_answer": self.correct_answer,
            "final_student_answer": self.final_student_answer,
            "correct": self.correct,
            "leaked_answer": self.leaked_answer,
            "parallel_thinking_count": self.parallel_thinking_count,
            "thinking_paths_count": self.thinking_paths_count,
            "total_turns": self.total_turns,
            "turns": self.turns
        }
        
        # 添加进步分析数据（如果有）
        if self.progress_analysis:
            data["progress_analysis"] = self.progress_analysis
            
        return data

    def _extract_errors(self):  # dddd  LLM extract
        """从对话记录中提取错误模式"""
        errors = []

        if not self.turns:
            return errors

        # 分析教师回复中的纠正信息
        for turn in self.turns:
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

class ExperimentRecorder:
    """实验记录器"""

    def __init__(self, experiment_name: str = "multi_agent_math"):
        self.experiment_name = experiment_name
        self.records = []
        self.summary_stats = {}
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    def add_record(self, record: DialogueRecord):
        """添加对话记录"""
        self.records.append(record)

    def calculate_statistics(self):
        """计算统计信息"""
        if not self.records:
            return {}

        total_problems = len(self.records)
        correct_answers = sum(1 for r in self.records if r.correct)
        leaked_answers = sum(1 for r in self.records if r.leaked_answer)
        total_turns = sum(r.total_turns for r in self.records)
        total_parallel_thinking = sum(r.parallel_thinking_count for r in self.records)
        total_thinking_paths = sum(r.thinking_paths_count for r in self.records)

        self.summary_stats = {
            "total_problems": total_problems,
            "accuracy": correct_answers / total_problems if total_problems > 0 else 0,
            "answer_leakage_rate": leaked_answers / total_problems if total_problems > 0 else 0,
            "avg_turns_per_problem": total_turns / total_problems if total_problems > 0 else 0,
            "avg_parallel_thinking": total_parallel_thinking / total_problems if total_problems > 0 else 0,
            "avg_thinking_paths": total_thinking_paths / total_problems if total_problems > 0 else 0,
            "correct_answers": correct_answers,
            "leaked_answers": leaked_answers
        }

        return self.summary_stats

    def save_results(self, output_dir: str = "results"):
        """保存结果到文件"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # 保存详细记录
        detailed_data = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "records": [record.to_dict() for record in self.records],
            "summary": self.summary_stats
        }

        return detailed_data, self.summary_stats

    def print_summary(self):
        """打印实验摘要"""
        if not self.summary_stats:
            self.calculate_statistics()

        print("\n" + "=" * 60)
        print("🎯 实验摘要统计")
        print("=" * 60)
        print(f"总问题数: {self.summary_stats['total_problems']}")
        print(f"准确率: {self.summary_stats['accuracy']:.4f}")
        print(f"答案泄露率: {self.summary_stats['answer_leakage_rate']:.4f}")
        print(f"平均对话轮数: {self.summary_stats['avg_turns_per_problem']:.2f}")
        print(f"平均并行思考次数: {self.summary_stats['avg_parallel_thinking']:.2f}")
        print(f"平均思考路径数: {self.summary_stats['avg_thinking_paths']:.2f}")
        print(f"正确答案数: {self.summary_stats['correct_answers']}")
        print(f"泄露答案数: {self.summary_stats['leaked_answers']}")
        print("=" * 60)
