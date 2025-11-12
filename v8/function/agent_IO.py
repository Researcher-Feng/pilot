import os
import sys
from typing import Callable, List, Optional, Dict, Any
from types import SimpleNamespace
sys.path.append(f'../..')
sys.path.append(rf'D:\DeepLearning\Code\LangChain')
sys.path.append(rf'/mnt/t2-6tb/medical/SocraticLM_langchain/LangChain_3090')
from v8.function.agent_core import SimpleAgent, ExpertStudentAgent
from v8.function.record_eval import DialogueRecord, ExperimentRecorder
from v8.function.memory import SmartSummaryMemory, SummaryConfig
from v8.function.cognitive import StudentCognitiveState
from v8.function.solution_tree import SolutionTree

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



