import datetime
from typing import Dict, Any
from v6.prompt.dialogue_cognitive import STUDENT_COGNITIVE_STATE_PROMPT, STUDENT_COGNITIVE_STATE_PROMPT_2

class StudentCognitiveState:
    """学生认知状态管理"""

    def __init__(self,
                 carelessness_level=5,
                 math_background="intermediate",
                 response_style="thoughtful",
                 preferred_method="balanced",
                 learning_style="visual"):

        self.carelessness_level = carelessness_level  # 1-10
        self.math_background = math_background  # beginner/intermediate/advanced
        self.response_style = response_style  # thoughtful/impulsive/detailed/brief
        self.preferred_method = preferred_method  # algebraic/geometric/computational/balanced
        self.learning_style = learning_style  # visual/auditory/kinesthetic/reading-writing

        # 动态属性
        self.problem_solving_history = []
        self.error_patterns = []
        self.method_preferences = []
        self.conceptual_understanding = {}

    def update_based_on_interaction(self, problem, student_approach, errors, method_used, success):
        """基于交互更新认知状态"""
        # 记录问题解决历史
        self.problem_solving_history.append({
            "problem": problem,
            "approach": student_approach,
            "errors": errors,
            "method_used": method_used,
            "success": success,
            "timestamp": datetime.datetime.now().isoformat()
        })

        # 更新粗心程度（基于错误模式）
        if errors:
            self.error_patterns.extend(errors)
            # 如果频繁出现计算错误，增加粗心程度
            calculation_errors = sum(1 for error in errors if "calculation" in error.lower())
            if calculation_errors > 0:
                self.carelessness_level = min(10, self.carelessness_level + 0.5)

        # 更新方法偏好
        self.method_preferences.append(method_used)

        # 更新数学背景（基于成功率和方法复杂度）
        if success and len(student_approach) > 3:  # 复杂的成功解法
            if self.math_background == "beginner":
                self.math_background = "intermediate"
            elif self.math_background == "intermediate" and len(self.problem_solving_history) > 5:
                self.math_background = "advanced"

    def to_dict(self):
        """转换为字典"""
        return {
            "carelessness_level": self.carelessness_level,
            "math_background": self.math_background,
            "response_style": self.response_style,
            "preferred_method": self.preferred_method,
            "learning_style": self.learning_style,
            "problem_solving_count": len(self.problem_solving_history),
            "recent_success_rate": self._calculate_recent_success_rate()
        }

    def _calculate_recent_success_rate(self):
        """计算最近的成功率"""
        if len(self.problem_solving_history) == 0:
            return 0.0
        recent = self.problem_solving_history[-5:]  # 最近5个问题
        successes = sum(1 for record in recent if record["success"])
        return successes / len(recent)

    def get_prompt_context(self):
        """获取提示词上下文"""
        return STUDENT_COGNITIVE_STATE_PROMPT_2.format(
            carelessness_level=self.carelessness_level,
            math_background=self.math_background,
            response_style=self.response_style,
            preferred_method=self.preferred_method,
            learning_style=self.learning_style
        )
