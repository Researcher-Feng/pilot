import os
import sys
from typing import Callable, List, Optional, Dict, Any
from types import SimpleNamespace
sys.path.append(f'../..')
sys.path.append(rf'D:\DeepLearning\Code\LangChain')
sys.path.append(rf'/mnt/t2-6tb/medical/SocraticLM_langchain/LangChain_3090')
from v7.function.agent_core import SimpleAgent, ExpertStudentAgent
from v7.function.record_eval import DialogueRecord, ExperimentRecorder
from v7.function.memory import SmartSummaryMemory, SummaryConfig
from v7.function.cognitive import StudentCognitiveState
from v7.function.solution_tree import SolutionTree
from v7.function.middleware import CustomMiddleware, MiddlewareFunc, ModelConfig
from v7.prompt.system import STUDENT_PROMPT_EASY, TEACHER_PROMPT_EASY, TEACHER_PROMPT, STUDENT_PROMPT
from v7.prompt.dialogue_cognitive import STUDENT_COGNITIVE_STATE_PROMPT
from v7.prompt.dialogue_tree_parallel import TREE_GENERATE_PROMPT, TEACHER_WITH_TREE_PROMPT
from v7.function.contex_fun import Context
from v7.function.format_fun import ResponseFormat
from v7.function.memory_fun import simple_checkpointer
from v7.utils.evaluator import extract_answer, logger
from v7.utils.MARIO_EVAL.demo import is_equiv_MATH

# Re-export main classes
__all__ = [
    'SimpleAgent',
    'ExpertStudentAgent',
    'DialogueRecord',
    'ExperimentRecorder',
    'SmartSummaryMemory',
    'StudentCognitiveState',
    'SolutionTree'
]



