import datetime
from typing import Dict, Any, List

class SolutionTree:
    """解题树管理"""

    def __init__(self, problem_statement):
        self.problem_statement = problem_statement
        self.root_node = {"type": "problem", "content": problem_statement}
        self.solution_paths = []
        self.current_student_path = []

    def add_expert_path(self, path_data):
        """添加专家解题路径"""
        self.solution_paths.append({
            "type": "expert",
            "path_id": len(self.solution_paths) + 1,
            "method": path_data.get("method", "unknown"),
            "complexity": path_data.get("complexity", "medium"),
            "innovation": path_data.get("innovation", "medium"),
            "steps": path_data.get("steps", []),
            "intermediate_answers": path_data.get("intermediate_answers", []),
            "final_answer": path_data.get("final_answer", "")
        })

    def add_student_step(self, step_content, method_used=None):
        """添加学生解题步骤"""
        # 如果没有提供方法，尝试从步骤内容中检测
        if method_used is None:
            method_used = self._detect_student_method(step_content)

        self.current_student_path.append({
            "step_number": len(self.current_student_path) + 1,
            "content": step_content,
            "method": method_used,
            "timestamp": datetime.datetime.now().isoformat()
        })

    def complete_student_path(self, success, final_answer=None):
        """完成学生解题路径"""
        student_path = {
            "type": "student",
            "path_id": f"student_{len(self.solution_paths) + 1}",
            "method": self._detect_student_method(),
            "steps": self.current_student_path.copy(),
            "success": success,
            "final_answer": final_answer
        }
        self.solution_paths.append(student_path)
        self.current_student_path = []

        return student_path

    def _detect_student_method(self, response=None):
        """检测学生使用的方法
        Args:
            response: 可选的响应文本，如果不提供则使用当前学生路径
        """
        if response:
            # 如果提供了响应文本，直接分析该文本
            content = response.lower()
        elif self.current_student_path:
            # 否则使用当前学生路径的所有内容
            content = " ".join([step["content"].lower() for step in self.current_student_path])
        else:
            return "unknown"

        # if not self.current_student_path:
        #     return "unknown"
        #
        # content = " ".join([step["content"].lower() for step in self.current_student_path])

        if any(word in content for word in ["equation", "solve for", "variable", "x ="]):
            return "algebraic"
        elif any(word in content for word in ["diagram", "graph", "shape", "angle", "area"]):
            return "geometric"
        elif any(word in content for word in ["calculate", "compute", "number", "digit"]):
            return "computational"
        elif any(word in content for word in ["logic", "reason", "therefore", "because", "since", "if then"]):
            return "logical"
        elif any(word in content for word in ["guess", "try", "maybe", "perhaps", "i think"]):
            return "trial_and_error"
        else:
            return "unknown"

    def compare_with_expert(self):
        """比较学生路径与专家路径"""
        if not self.current_student_path:
            return {"similarity": 0, "closest_expert_path": None}

        student_method = self._detect_student_method()
        expert_paths = [p for p in self.solution_paths if p["type"] == "expert"]

        if not expert_paths:
            return {"similarity": 0, "closest_expert_path": None}

        # 简单的方法匹配
        matching_paths = [p for p in expert_paths if p["method"] == student_method]
        if matching_paths:
            closest = matching_paths[0]
            similarity = 0.8  # 方法匹配的基础相似度
        else:
            closest = expert_paths[0]
            similarity = 0.3  # 方法不匹配的基础相似度

        return {
            "similarity": similarity,
            "closest_expert_path": closest,
            "method_match": student_method == closest["method"]
        }

    def to_dict(self):
        """转换为字典"""
        return {
            "problem_statement": self.problem_statement,
            "solution_paths": self.solution_paths,
            "current_student_path": self.current_student_path
        }
