import os
import sys
from typing import Callable, List, Optional, Dict, Any
from types import SimpleNamespace

sys.path.append(f'../..')
sys.path.append(rf'D:\DeepLearning\Code\LangChain')
sys.path.append(rf'/mnt/t2-6tb/medical/SocraticLM_langchain/LangChain_3090')
from v6.function.agent_core import SimpleAgent, LocalModelWrapper, ExpertStudentAgent
from v6.function.record_eval import DialogueRecord, ExperimentRecorder
from v6.function.memory import SmartSummaryMemory, SummaryConfig
from v6.function.cognitive import StudentCognitiveState
from v6.function.solution_tree import SolutionTree
from v6.function.middleware import CustomMiddleware, MiddlewareFunc, ModelConfig
from v6.prompt.system import STUDENT_PROMPT_EASY, TEACHER_PROMPT_EASY, TEACHER_PROMPT, STUDENT_PROMPT
from v6.prompt.dialogue_cognitive import STUDENT_COGNITIVE_STATE_PROMPT
from v6.prompt.dialogue_tree_parallel import TREE_GENERATE_PROMPT, TEACHER_WITH_TREE_PROMPT
from v6.function.contex_fun import Context
from v6.function.format_fun import ResponseFormat
from v6.function.memory_fun import simple_checkpointer
from config_secret.api_config import ds_key_url, key_config, key_url, smith_key
from v6.utils.evaluator import extract_answer, logger
from v6.utils.MARIO_EVAL.demo import is_equiv_MATH

selected_model = 'deepseek-official'
os.environ["OPENAI_API_KEY"] = key_config.get(selected_model, "")
os.environ["OPENAI_BASE_URL"] = key_url
os.environ["DEEPSEEK_API_KEY"] = key_config.get(selected_model, "")
os.environ["DEEPSEEK_BASE_URL"] = ds_key_url
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_API_KEY"] = smith_key
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('ALL_PROXY', None)
os.environ.pop('all_proxy', None)

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



