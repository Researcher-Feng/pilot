
# ... (top of the file remains unchanged) ...

import graphviz  # Added for tree visualization

# ... (rest of the imports and class definitions remain unchanged) ...

# In the solve_with_dialogue function, inside the loop after solving the problem:
if config.agent.get("use_solution_tree", False) and multi_agent_system.current_solution_tree:
    solution_tree = multi_agent_system.current_solution_tree
    logger.info(f"🌳 解题树统计:")
    logger.info(f"   专家路径数: {len([p for p in solution_tree.solution_paths if p.get('type') == 'expert'])}")
    logger.info(f"   学生路径数: {len([p for p in solution_tree.solution_paths if p.get('type') == 'student'])}")

    # Added: Visualize trees in debug mode
    if dialogue_record.debug_mode:
        logger.info("🌳 Visualizing Expert Tree:")
        vis_result = solution_tree.visualize_graph(f'expert_tree_{i}', owner='expert')
        logger.info(vis_result)

        logger.info("🌳 Visualizing Student Tree:")
        vis_result = solution_tree.visualize_graph(f'student_tree_{i}', owner='student')
        logger.info(vis_result)

    # ... (rest of the multi-solution scoring and logging remains unchanged) ...

# ... (rest of the file remains unchanged) ...



