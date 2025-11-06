# 在 multi_agent_chat_explicit 方法中，确保正确调用
# 学生回应后记录步骤
student_response_obj = self._invoke_agent(student_input)
student_response = self._extract_response_text(student_response_obj)
self.dialogue_history.append(("student", student_response))

# 记录学生解题步骤
if self.use_solution_tree and self.solution_tree:
    method_used = self._detect_method_from_response(student_response)
    self.record_student_step(student_response, method_used=method_used)  # 调用 SimpleAgent 的方法
    if dialogue_record.debug_mode:
        logger.info(f"📝 已记录学生第{current_turn}步解题步骤，方法: {method_used}")

