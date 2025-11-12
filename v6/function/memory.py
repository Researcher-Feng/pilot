import datetime
from typing import Optional, Dict, Any

class SmartSummaryMemory:
    """智能对话摘要内存管理"""

    def __init__(self, llm=None, max_turns=10, max_token_limit=2000, enabled=False, debug_mode=False):
        self.debug_mode = debug_mode
        self.enabled = enabled
        self.max_turns = max_turns
        self.max_token_limit = max_token_limit
        self.llm = llm
        self.conversation_history = []
        self.summary_history = []  # 存储历史摘要
        self.turn_count = 0
        self.current_summary = ""

    def add_message(self, role: str, content: str):
        """添加消息到对话历史"""
        if not self.enabled:
            return

        self.conversation_history.append({"role": role, "content": content})
        self.turn_count += 1

        # 检查是否需要生成摘要
        if self._should_generate_summary():
            self._generate_summary()

    def _should_generate_summary(self):
        """判断是否需要生成摘要"""
        if not self.enabled:
            return False

        # 基于轮次判断
        if self.turn_count >= self.max_turns:
            return True

        # 基于token数量判断（简单估算）
        total_chars = sum(len(msg["content"]) for msg in self.conversation_history)
        estimated_tokens = total_chars // 4  # 简单估算：1 token ≈ 4 characters
        if estimated_tokens > self.max_token_limit:
            return True

        return False

    def _generate_summary(self):
        """生成对话摘要"""
        if not self.enabled or not self.llm:
            return

        try:
            # 构建摘要提示词
            conversation_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in self.conversation_history]
            )

            summary_prompt = f"""Please summarize the key information from this math tutoring conversation, focusing on:

1. The main math problem being discussed
2. Student's current approach and difficulties
3. Teacher's guidance and key questions asked
4. Important intermediate steps and concepts
5. Current progress toward solution

Conversation:
{conversation_text}

Provide a concise but informative summary in English:"""

            # 调用LLM生成摘要
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(summary_prompt)
                summary = response.content if hasattr(response, 'content') else str(response)
            else:
                # 简化调用
                summary = "Summary: Math problem discussion in progress"

            self.current_summary = summary
            self.summary_history.append({
                "turn_count": self.turn_count,
                "summary": summary,
                "timestamp": datetime.datetime.now().isoformat()
            })

            # 保留最近对话作为上下文
            keep_recent = min(3, len(self.conversation_history))
            self.conversation_history = self.conversation_history[-keep_recent:]
            self.turn_count = keep_recent

            if self.debug_mode:
                logger.info(f"📝 已生成对话摘要 (第{len(self.summary_history)}次摘要)")
                logger.info(f"Summary: {summary[:200]}...")

        except Exception as e:
            logger.warning(f"❌ 生成摘要时出错: {e}")
            self.current_summary = f"Conversation history: {len(self.conversation_history)} turns about math problem"

    def get_context(self):
        """获取当前上下文（包含摘要）"""
        if not self.enabled:
            return ""

        context_parts = []

        # 添加当前摘要
        if self.current_summary:
            context_parts.append(f"Previous Conversation Summary:\n{self.current_summary}")

        # 添加最近对话历史
        recent_messages = self.conversation_history[-3:]  # 保留最近3轮
        if recent_messages:
            recent_text = "\n".join(
                [f"{msg['role']}: {msg['content']}" for msg in recent_messages]
            )
            context_parts.append(f"Recent Dialogue:\n{recent_text}")

        return "\n\n".join(context_parts) if context_parts else ""

    def clear(self):
        """清空内存"""
        self.conversation_history.clear()
        self.summary_history.clear()
        self.turn_count = 0
        self.current_summary = ""

    def get_stats(self):
        """获取内存统计信息"""
        return {
            "enabled": self.enabled,
            "total_turns": self.turn_count,
            "conversation_history_count": len(self.conversation_history),
            "summary_history_count": len(self.summary_history),
            "current_summary_length": len(self.current_summary)
        }


class SummaryConfig:
    """摘要配置类"""
    def __init__(self, enabled=False, max_turns=10, max_token_limit=2000, summary_model=None):
        self.enabled = enabled
        self.max_turns = max_turns
        self.max_token_limit = max_token_limit
        self.summary_model = summary_model  # 专门用于摘要的模型配置


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

        return leakage_detected

    def to_dict(self):
        """转换为字典格式"""
        return {
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
