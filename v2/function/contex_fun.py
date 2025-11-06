# contex_fun.py
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Context:
    """增强的运行时上下文schema"""
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    math_background: Optional[str] = "intermediate"  # beginner, intermediate, advanced
    parallel_thinking: bool = False
    socratic_teaching: bool = False
    dialogue_turn: int = 0
    current_problem: Optional[str] = None
    user_preferences: Optional[Dict[str, Any]] = None

    def update_turn(self):
        """更新对话轮次"""
        self.dialogue_turn += 1

    def reset_dialogue(self):
        """重置对话状态"""
        self.dialogue_turn = 0
        self.current_problem = None