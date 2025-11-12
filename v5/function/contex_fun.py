# contex_fun.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, List


@dataclass
class Context:
    """Enhanced runtime context schema for multi-agent math tutoring"""
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    math_background: Optional[str] = None  # beginner, intermediate, advanced
    parallel_thinking: bool = False
    socratic_teaching: bool = False
    dialogue_turn: int = 0
    current_problem: Optional[str] = None
    user_preferences: Optional[Dict[str, Any]] = None
    learning_style: Optional[str] = "visual"  # visual, auditory, kinesthetic
    difficulty_level: Optional[str] = "medium"  # easy, medium, hard
    conversation_mode: Optional[str] = "explicit"  # explicit, tool_based

    def update_turn(self):
        """Update dialogue turn count"""
        self.dialogue_turn += 1

    def reset_dialogue(self):
        """Reset dialogue state"""
        self.dialogue_turn = 0
        self.current_problem = None

    def set_math_background(self, level: str):
        """Set math background level"""
        valid_levels = ["beginner", "intermediate", "advanced"]
        if level in valid_levels:
            self.math_background = level

    def enable_parallel_thinking(self, enabled: bool = True):
        """Enable or disable parallel thinking"""
        self.parallel_thinking = enabled

    def enable_socratic_teaching(self, enabled: bool = True):
        """Enable or disable Socratic teaching"""
        self.socratic_teaching = enabled

    def set_conversation_mode(self, mode: str):
        """Set conversation mode"""
        valid_modes = ["explicit", "tool_based"]
        if mode in valid_modes:
            self.conversation_mode = mode

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            "user_id": self.user_id,
            "user_role": self.user_role,
            "math_background": self.math_background,
            "parallel_thinking": self.parallel_thinking,
            "socratic_teaching": self.socratic_teaching,
            "dialogue_turn": self.dialogue_turn,
            "current_problem": self.current_problem,
            "learning_style": self.learning_style,
            "difficulty_level": self.difficulty_level,
            "conversation_mode": self.conversation_mode
        }