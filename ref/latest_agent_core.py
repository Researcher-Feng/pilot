
# Latest: function/agent_core.py
# Changes:
# - Made _parse_solution_tree staticmethod for easy call from solution_tree.py.
# - In record_student_step: Pass raw_problem and solution_tree_gold_section if available (assume from self).
# - In multi_agent_chat_explicit: Unchanged.
# - In generate_solution_tree: Minor.

from typing import Dict, Any, Optional
from types import SimpleNamespace

from v9.utils.evaluator import logger, extract_answer
from v9.function.api_client import create_api_client, RawAPIClient, RawOllamaClient
from v9.utils.MARIO_EVAL.demo import is_equiv_MATH
from v9.function.memory import SmartSummaryMemory, SummaryConfig
from v9.function.solution_tree import SolutionTree  # Updated import
from v9.function.cognitive import StudentCognitiveState
from v9.function.middleware import CustomMiddleware, ModelConfig, MiddlewareFunc
from v9.function.contex_fun import Context
from v9.function.format_fun import ResponseFormat
from v9.function.record_eval import DialogueRecord
from v9.prompt.system import *
from v9.prompt.dialogue_tree_parallel import *
from v9.prompt.dialogue_socratic import *
import re  # For parsing

class SimpleLocalAgent:
    """简化版本地Agent，支持核心功能"""  # (Unchanged)

    # ... (existing code)

class SimpleAgent(object):
    """增强的Agent类，支持多模式和本地调用"""

    def __init__(self, agent_type: str = "student", debug_mode: bool = False):
        # ... (existing)
        self.is_first_student_response = True  # New flag for initial tree build

    # ... (existing set_cognitive_state, set_solution_tree, etc.)

    def trigger_parallel_thinking(self, impasse_detected: bool, N: int = 3):
        """Trigger parallel thinking: Branch tree and verify."""
        if not impasse_detected or not self.use_solution_tree or not self.solution_tree:
            return

        # Get seeds from Xueba if available
        if hasattr(self, 'expert_agent'):  # Assuming access or pass as param
            seeds = self.expert_agent.seed_parallel_paths(N)
        else:
            seeds = [f"Seed path {i}" for i in range(N)]  # Dummy

        # Find branch point (e.g., last student node or impasse node)
        branch_from = self.solution_tree.current_student_path
            [-1] if self.solution_tree.current_student_path else self.solution_tree.root_node_id

        for seed in seeds:
            self.solution_tree.parse_and_add_student_response(seed, model=self.model, add_as_branch_from=branch_from, raw_problem=self.problem_statement, solution_tree_gold_section=self.solution_tree_gold_section)

        # Verify and summarize
        self.solution_tree.verify_paths_via_contradiction(model=self.model)  # Pass model for prompting
        summary = self.solution_tree.summarize_paths()
        logger.info(f"Parallel thinking summary: {summary}")

    def _build_full_prompt(self, user_input, memory_context, prompt_type='api'):
        """构建完整提示词（增强版）"""  # (Unchanged)

        # ... (existing)

    def record_student_step(self, step_content: str, method_used: Optional[str] = None):
        """Wrapper to add step and potentially trigger parallel."""
        if self.use_solution_tree and self.solution_tree:
            # Use expert model if available for refinement
            refine_model = self.expert_agent.model if hasattr(self, 'expert_agent') else self.model
            branch_from = None if self.is_first_student_response else self.solution_tree.current_student_path
                [-1] if self.solution_tree.current_student_path else None
            self.solution_tree.parse_and_add_student_response(step_content, method_used, model=refine_model, add_as_branch_from=branch_from, raw_problem=self.problem_statement, solution_tree_gold_section=self.solution_tree_gold_section)
            self.is_first_student_response = False  # Set after first

        # Check for impasse (simple heuristic or prompt model)
        impasse = "stuck" in step_content.lower()  # Dummy; improve with model query
        self.trigger_parallel_thinking(impasse)

    def complete_student_solution(self, success: bool, final_answer: Optional[str] = None):
        """Complete and update tree."""
        if self.use_solution_tree and self.solution_tree:
            self.solution_tree.complete_student_path(success, final_answer)
            self.solution_tree.prune_invalid_paths()

    # ... (existing _format_teacher_input, etc.)

    def analyze_student_tree(self) -> str:
        """Analyze student tree for teacher."""
        if not self.use_solution_tree or not self.solution_tree:
            return ""

        comp = self.solution_tree.compare_with_expert()
        blind_spots = []  # Logic to detect: e.g., missing methods from gold
        for gold in self.solution_tree.gold_paths:
            if gold["method"] not in [self.solution_tree.nodes[nid]["edge_method"] for nid in self.solution_tree.nodes if nid != "root" and self.solution_tree.nodes[nid].get("owner") == "student"]:
                blind_spots.append(gold["method"])

        error_causes = []  # From invalid nodes
        for nid, node in self.solution_tree.nodes.items():
            if not node["is_valid"] and node.get("owner") == "student":
                error_causes.append(f"Node {nid}: {node['content'][:20]}... invalid due to contradiction.")

        return f"Similarity to gold: {comp['similarity']}. Blind spots: {', '.join(blind_spots)}. Errors: {', '.join(error_causes)}."

    def multi_agent_chat_explicit(self, problem, correct_answer):
        """Explicit chat loop with interventions."""
        self.problem_statement = problem
        self.correct_answer = correct_answer
        self.dialogue_history = []
        self.current_turn = 0
        self.student_answer = None

        # Initial student response
        student_prompt = f"Solve: {problem}"
        student_response = self._invoke_student(student_prompt)  # Assume method to invoke student
        self.record_student_step(student_response)
        self.dialogue_history.append(("student", student_response))
        self.student_answer = extract_answer(student_response)  # Assume function

        while self.current_turn < self.max_turns and not self._has_correct_answer(self.student_answer, correct_answer):
            analysis = self.analyze_student_tree()
            teacher_prompt = f"Student response: {student_response}\nBased on analysis: {analysis}, guide student without giving answer."
            teacher_response = self._invoke_teacher(teacher_prompt)  # Assume method
            self.dialogue_history.append(("teacher", teacher_response))

            # Next student response based on guidance
            student_prompt = f"Teacher guidance: {teacher_response}\nContinue solving: {problem}"
            student_response = self._invoke_student(student_prompt)
            self.record_student_step(student_response)
            self.dialogue_history.append(("student", student_response))
            self.student_answer = extract_answer(student_response)

            self.current_turn += 1

        # Complete
        success = self._has_correct_answer(self.student_answer, correct_answer)
        self.complete_student_solution(success, self.student_answer)
        return self.dialogue_history

    def _invoke_student(self, prompt: str):
        """Stub for student invoke."""
        messages = [{"role": "user", "content": prompt}]
        return self.model.invoke(messages).content  # Simplified

    def _invoke_teacher(self, prompt: str):
        """Stub for teacher invoke."""
        messages = [{"role": "user", "content": prompt}]
        return self.model.invoke(messages).content  # Simplified; use teacher model

# For ExpertStudentAgent
class ExpertStudentAgent(SimpleAgent):
    # ... (existing)

    def generate_solution_tree(self, problem):
        """Generate tree with weights."""  # Enhanced
        prompt = TREE_GENERATE_PROMPT.format(problem) + "\nInclude weights for complexity and innovation in each path."
        response = self._invoke_agent(prompt)
        return self._parse_solution_tree_static(response, problem)  # Use static

    @staticmethod
    def _parse_solution_tree_static(response, problem):
        """Static version of parse for external call."""
        solution_tree = SolutionTree(problem)

        try:
            # 简单的解析逻辑 - 在实际应用中可以使用更复杂的解析
            if "<SolutionTree>" in response:
                # 提取解决方案路径
                paths_section = response.split("<SolutionPaths>")[1].split("</SolutionPaths>")[0]
                path_blocks = paths_section.split("</Path>")

                for block in path_blocks:
                    if "<Path" in block:
                        # 提取路径信息
                        method = ExpertStudentAgent._extract_site_tag_static(block, "method")
                        complexity = ExpertStudentAgent._extract_site_tag_static(block, "complexity")
                        innovation = ExpertStudentAgent._extract_site_tag_static(block, "innovation")

                        # 提取步骤
                        steps = []
                        intermediate_answers = []
                        step_parts = block.split("<Step")
                        for step_part in step_parts[1:]:
                            if ">" in step_part and "</Step>" in step_part:
                                step_content = step_part.split(">", 1)[1].split("</Step>")[0]
                                steps.append(step_content)
                            if "<IntermediateAnswer>" in step_part and "</IntermediateAnswer>" in step_part:
                                intermediate_content = \
                                    step_part.split("<IntermediateAnswer>", 1)[1].split("</IntermediateAnswer>")[0]
                                intermediate_answers.append(intermediate_content)

                        # 提取最终答案
                        final_answer = ExpertStudentAgent._extract_xml_tag_static(block, "FinalAnswer")

                        solution_tree.add_expert_path({
                            "method": method,
                            "complexity": complexity,
                            "innovation": innovation,
                            "steps": steps,
                            "intermediate_answers": intermediate_answers,
                            "final_answer": final_answer
                        })

        except Exception as e:
            logger.error(f"❌ Error parsing solution tree: {e}")
            # 如果解析失败，创建一个默认的解决方案路径
            solution_tree.add_expert_path({
                "method": "algebraic",
                "complexity": "medium",
                "innovation": "medium",
                "steps": ["Apply standard algebraic approach", "Solve step by step"],
                "intermediate_answers": [],
                "final_answer": "[[Answer will be determined]]"
            })

        return solution_tree

    @staticmethod
    def _extract_xml_tag_static(text, tag_name):
        """Static extract XML tag."""
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"

        if start_tag in text and end_tag in text:
            return text.split(start_tag)[1].split(end_tag)[0].strip()
        return ""

    @staticmethod
    def _extract_site_tag_static(text, tag_name):
        """Static extract attribute."""
        start_tag = f'{tag_name}="'
        end_tag = f'"'

        if start_tag in text and end_tag in text:
            return text.split(start_tag)[1].split(end_tag)[0].strip()
        return ""

    def seed_parallel_paths(self, N: int) -> List[str]:
        """Generate N diverse path seeds."""
        prompt = f"Generate {N} diverse starting ideas for problem: {self.problem_statement}. Format as <Idea>idea text</Idea> for each."
        response = self._invoke_agent(prompt)
        # Parse: Extract <Idea> tags
        ideas = re.findall(r'<Idea>(.*?)</Idea>', response, re.DOTALL)
        return [idea.strip() for idea in ideas] or [f"Default idea {i}" for i in range(N)]

    # ... (existing)



