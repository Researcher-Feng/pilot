# test_modules.py
"""
多智能体数学辅导系统测试模块
"""

import unittest
import tempfile
import os
import sys

# 添加路径
sys.path.append('../..')
sys.path.append('..')

from agent_IO import DialogueRecord, ExperimentRecorder, ModelConfig, SimpleAgent
from contex_fun import Context
from tools_fun import parallel_thinking, socratic_questioning, math_concept_explainer


class TestDialogueRecord(unittest.TestCase):
    """测试对话记录类"""

    def setUp(self):
        self.record = DialogueRecord("Test problem", "42")

    def test_initialization(self):
        self.assertEqual(self.record.problem, "Test problem")
        self.assertEqual(self.record.correct_answer, "42")
        self.assertEqual(self.record.turns, [])
        self.assertFalse(self.record.correct)
        self.assertFalse(self.record.leaked_answer)

    def test_add_turn(self):
        turn_data = {
            "turn": 1,
            "student_response": "I think it's 40",
            "teacher_response": "Close, try again",
            "teacher_intent": "encouragement"
        }
        self.record.add_turn(turn_data)

        self.assertEqual(len(self.record.turns), 1)
        self.assertEqual(self.record.total_turns, 1)
        self.assertEqual(self.record.turns[0]["turn"], 1)

    def test_analyze_student_response(self):
        response = "Let me think <Parallel>first approach</Parallel> and <Path>alternative</Path>"
        parallel_count, path_count = self.record.analyze_student_response(response)

        self.assertEqual(parallel_count, 1)
        self.assertEqual(path_count, 1)
        self.assertEqual(self.record.parallel_thinking_count, 1)
        self.assertEqual(self.record.thinking_paths_count, 1)

    def test_check_answer_leakage(self):
        # 测试答案泄露检测
        leaked = self.record.check_answer_leakage("The answer is 42")
        self.assertTrue(leaked)
        self.assertTrue(self.record.leaked_answer)

        # 测试无泄露
        record2 = DialogueRecord("Test", "42")
        not_leaked = record2.check_answer_leakage("Think about it")
        self.assertFalse(not_leaked)
        self.assertFalse(record2.leaked_answer)


class TestExperimentRecorder(unittest.TestCase):
    """测试实验记录器"""

    def setUp(self):
        self.recorder = ExperimentRecorder("test_experiment")

    def test_initialization(self):
        self.assertEqual(self.recorder.experiment_name, "test_experiment")
        self.assertEqual(self.recorder.records, [])
        self.assertEqual(self.recorder.summary_stats, {})

    def test_add_records_and_calculate_stats(self):
        # 添加测试记录
        for i in range(3):
            record = DialogueRecord(f"Problem {i}", f"Answer {i}")
            record.correct = (i % 2 == 0)  # 第一个和第三个正确
            record.parallel_thinking_count = i + 1
            record.thinking_paths_count = i + 2
            record.total_turns = i + 1
            self.recorder.add_record(record)

        stats = self.recorder.calculate_statistics()

        self.assertEqual(stats["total_problems"], 3)
        self.assertAlmostEqual(stats["accuracy"], 2 / 3)  # 2 out of 3 correct
        self.assertEqual(stats["avg_parallel_thinking"], (1 + 2 + 3) / 3)
        self.assertEqual(stats["avg_thinking_paths"], (2 + 3 + 4) / 3)
        self.assertEqual(stats["avg_turns_per_problem"], (1 + 2 + 3) / 3)

    def test_save_results(self):
        # 添加一个测试记录
        record = DialogueRecord("Test problem", "42")
        record.correct = True
        self.recorder.add_record(record)

        # 使用临时目录测试保存功能
        with tempfile.TemporaryDirectory() as temp_dir:
            detailed_file, summary_file = self.recorder.save_results(temp_dir)

            self.assertTrue(os.path.exists(detailed_file))
            self.assertTrue(os.path.exists(summary_file))

            # 检查文件内容
            import json
            with open(detailed_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.assertEqual(data["experiment_name"], "test_experiment")
                self.assertEqual(len(data["records"]), 1)


class TestModelConfig(unittest.TestCase):
    """测试模型配置类"""

    def test_api_config(self):
        config = ModelConfig(
            model_type="api",
            model_name="gpt-4",
            temperature=0.5,
            max_tokens=1000
        )

        self.assertEqual(config.model_type, "api")
        self.assertEqual(config.model_name, "gpt-4")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 1000)

    def test_local_config(self):
        config = ModelConfig(
            model_type="local",
            model_name="qwen2.5:0.5b",
            base_url="http://localhost:11434",
            temperature=0.7
        )

        self.assertEqual(config.model_type, "local")
        self.assertEqual(config.model_name, "qwen2.5:0.5b")
        self.assertEqual(config.base_url, "http://localhost:11434")
        self.assertEqual(config.temperature, 0.7)


class TestTools(unittest.TestCase):
    """测试工具函数"""

    def test_parallel_thinking(self):
        result = parallel_thinking.invoke("test problem")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_socratic_questioning(self):
        result = socratic_questioning.invoke("math problem")
        self.assertIsInstance(result, str)
        self.assertTrue("Socratic guidance" in result)

    def test_math_concept_explainer(self):
        # 测试代数概念解释
        result = math_concept_explainer.invoke({"concept": "algebra", "level": "beginner"})
        self.assertIsInstance(result, str)
        self.assertTrue("Algebra" in result)

        # 测试几何概念解释
        result = math_concept_explainer.invoke({"concept": "geometry", "level": "intermediate"})
        self.assertIsInstance(result, str)
        self.assertTrue("Geometry" in result)


class TestContext(unittest.TestCase):
    """测试上下文类"""

    def test_context_initialization(self):
        context = Context(
            user_id="test_user",
            user_role="student",
            math_background="beginner",
            parallel_thinking=True
        )

        self.assertEqual(context.user_id, "test_user")
        self.assertEqual(context.user_role, "student")
        self.assertEqual(context.math_background, "beginner")
        self.assertTrue(context.parallel_thinking)

    def test_context_methods(self):
        context = Context()

        # 测试设置数学背景
        context.set_math_background("advanced")
        self.assertEqual(context.math_background, "advanced")

        # 测试无效背景级别
        context.set_math_background("invalid")
        self.assertEqual(context.math_background, "advanced")  # 应该保持不变

        # 测试启用功能
        context.enable_parallel_thinking(True)
        self.assertTrue(context.parallel_thinking)

        context.enable_socratic_teaching(False)
        self.assertFalse(context.socratic_teaching)

        # 测试对话模式设置
        context.set_conversation_mode("tool_based")
        self.assertEqual(context.conversation_mode, "tool_based")

        context.set_conversation_mode("invalid")
        self.assertEqual(context.conversation_mode, "tool_based")  # 应该保持不变

    def test_to_dict(self):
        context = Context(
            user_id="test",
            user_role="teacher",
            math_background="intermediate"
        )

        context_dict = context.to_dict()

        self.assertEqual(context_dict["user_id"], "test")
        self.assertEqual(context_dict["user_role"], "teacher")
        self.assertEqual(context_dict["math_background"], "intermediate")
        self.assertIn("learning_style", context_dict)
        self.assertIn("difficulty_level", context_dict)


if __name__ == '__main__':
    # 运行所有测试
    unittest.main(verbosity=2)