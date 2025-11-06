# 在对话结束后，输出解题树信息
if config.agent.get("use_solution_tree", False) and multi_agent_system.current_solution_tree:
    solution_tree = multi_agent_system.current_solution_tree
    logger.info(f"🌳 解题树统计:")
    logger.info(f"   专家路径数: {len([p for p in solution_tree.solution_paths if p.get('type') == 'expert'])}")
    logger.info(f"   学生路径数: {len([p for p in solution_tree.solution_paths if p.get('type') == 'student'])}")

    # 输出学生路径详情
    student_paths = [p for p in solution_tree.solution_paths if p.get('type') == 'student']
    for i, path in enumerate(student_paths):
        logger.info(
            f"   学生路径 {i + 1}: 步骤数={len(path.get('steps', []))}, 成功={path.get('success', False)}, 方法={path.get('method', 'unknown')}")

