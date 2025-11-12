import datetime
import json
import re  # For parsing
from typing import Dict, Any, List, Optional
from v9.prompt.dialogue_tree_parallel import *  # Assume EXTRACT_TREE_GENERATE_PROMPT here
import graphviz  # Added for graphical visualization
import os
os.environ["PATH"] += os.pathsep + r'D:\Program\Graphviz\bin'

class SolutionTree:
    """解题树管理"""

    def __init__(self, problem_statement):
        self.problem_statement = problem_statement
        self.root_node_id = "root"
        self.nodes: Dict[str, Dict[str, Any]] = {
            self.root_node_id: {
                "type": "problem",
                "content": problem_statement,
                "children": [],
                "parent_id": None,
                "edge_method": None,
                "weight": {"complexity": 0.0, "innovation": 0.0},
                "is_valid": True,
                "timestamp": datetime.datetime.now().isoformat(),
                "owner": None
            }
        }
        self.next_node_id = 1  # For generating unique node IDs
        self.gold_paths: List[Dict] = []  # For gold standard paths
        self.current_student_path: List[str] = []  # List of node IDs in current path

    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        node_id = f"node_{self.next_node_id}"
        self.next_node_id += 1
        return node_id

    def add_branch(self, parent_id: str, method: str, content: str, is_leaf: bool = False,
                   intermediate_answer: Optional[str] = None, owner: Optional[str] = None) -> str:
        """Add a branch (child node) from a parent."""
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node {parent_id} does not exist.")

        node_id = self._generate_node_id()
        node_type = "leaf" if is_leaf else "intermediate"
        # Heuristic weights: complexity based on depth, innovation based on method rarity (dummy mapping for now)
        depth = self._get_depth(parent_id) + 1
        complexity = min(depth / 5.0, 1.0)  # Normalize to [0,1]
        innovation_map = {"algebraic": 0.5, "geometric": 0.6, "logical": 0.7, "equations": 0.4, "disproof": 0.8, "unknown": 0.3}
        innovation = innovation_map.get(method, 0.5)

        self.nodes[node_id] = {
            "type": node_type,
            "content": content,
            "children": [],
            "parent_id": parent_id,
            "edge_method": method,
            "weight": {"complexity": complexity, "innovation": innovation},
            "is_valid": True,
            "timestamp": datetime.datetime.now().isoformat(),
            "intermediate_answer": intermediate_answer,
            "owner": owner  # New field
        }
        self.nodes[parent_id]["children"].append(node_id)
        return node_id

    def add_expert_path(self, path_data: Dict):
        """Add an expert path (backward compatible: converts to nodes)."""
        current_id = self.root_node_id
        for i, step in enumerate(path_data.get("steps", [])):
            is_leaf = (i == len(path_data["steps"]) - 1)
            intermediate = path_data.get("intermediate_answers", [None] * len(path_data["steps"]))[i]
            current_id = self.add_branch(
                current_id,
                path_data.get("method", "unknown"),
                step,
                is_leaf=is_leaf,
                intermediate_answer=intermediate,
                owner="expert"  # Set owner
            )
        # Set final answer on leaf
        if current_id in self.nodes:
            self.nodes[current_id]["final_answer"] = path_data.get("final_answer", "")

    def parse_and_add_student_response(self, response: str, method_used: Optional[str] = None,
                                       model: Optional[Any] = None, add_as_branch_from: Optional[str] = None,
                                       raw_problem=None, solution_tree_gold_section=None):
        """Parse student response for multiple steps and add as sequential nodes. Use model to refine if unstructured."""
        # If adding to existing branch (subsequent responses)
        if add_as_branch_from:
            parent_id = add_as_branch_from
            self.current_student_path = [add_as_branch_from]  # Reset to branch point
        elif not self.current_student_path:
            parent_id = self.root_node_id
        else:
            parent_id = self.current_student_path[-1]

        # Check if structured
        steps = re.findall(r'<Step number="(\d+)">(.*?)</Step>', response, re.DOTALL)

        if not steps and model:
            # Use model (expert) to refine unstructured response
            refine_prompt = EXTRACT_TREE_GENERATE_PROMPT.format(question=raw_problem, response=response, GOLD=solution_tree_gold_section)
            refined = model.invoke([{"role": "user", "content": refine_prompt}])
            refined = refined['structured_response'].main_response

            # Now parse the refined string using _parse_solution_tree (made static)
            temp_tree, _ = self._parse_solution_tree(refined, raw_problem)  # Static call

            # Add parsed paths to main tree
            for path_data in temp_tree.solution_paths:
                current_id = parent_id
                for i, step in enumerate(path_data.get("steps", [])):
                    is_leaf = (i == len(path_data["steps"]) - 1)
                    intermediate = path_data.get("intermediate_answers", [None] * len(path_data["steps"]))[i]
                    step_method = method_used or path_data.get("method", "unknown")
                    current_id = self.add_branch(
                        current_id,
                        step_method,
                        step,
                        is_leaf=is_leaf,
                        intermediate_answer=intermediate,
                        owner="student"
                    )
                if current_id in self.nodes:
                    self.nodes[current_id]["final_answer"] = path_data.get("final_answer", "")
                self.current_student_path.extend(self._get_path_from(parent_id, current_id))  # Update path

    def _parse_solution_tree(self, response, problem):
        """解析解题树响应"""
        solution_tree = SolutionTree(problem)

        try:
            # 简单的解析逻辑 - 在实际应用中可以使用更复杂的解析
            if "<SolutionTree>" in response:

                paths_section = response.split("<SolutionPaths>")[1].split("</SolutionPaths>")[0]
                path_blocks = paths_section.split("</Path>")

                for block in path_blocks:
                    if "<Path" in block:
                        # 提取路径信息
                        method = self._extract_site_tag(block, "method")
                        complexity = self._extract_site_tag(block, "complexity")
                        innovation = self._extract_site_tag(block, "innovation")

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
                        final_answer = self._extract_xml_tag(block, "FinalAnswer")

                        solution_tree.add_expert_path({
                            "method": method,
                            "complexity": complexity,
                            "innovation": innovation,
                            "steps": steps,
                            "intermediate_answers": intermediate_answers,
                            "final_answer": final_answer
                        })

        except Exception as e:
            print(f"❌ Error parsing solution tree: {e}")
            # 如果解析失败，创建一个默认的解决方案路径
            solution_tree.add_expert_path({
                "method": "algebraic",
                "complexity": "medium",
                "innovation": "medium",
                "steps": ["Apply standard algebraic approach", "Solve step by step"],
                "intermediate_answers": [],
                "final_answer": "[[Answer will be determined]]"
            })

        return solution_tree, "<SolutionTree>\n<SolutionPaths>" + response.split("<SolutionPaths>")[1]

    def _extract_xml_tag(self, text, tag_name):
        """提取XML标签内容"""
        start_tag = f"<{tag_name}>"
        end_tag = f"</{tag_name}>"

        if start_tag in text and end_tag in text:
            return text.split(start_tag)[1].split(end_tag)[0].strip()
        return ""

    def _extract_site_tag(self, text, tag_name):
        """提取XML标签内容"""
        start_tag = f'{tag_name}="'
        end_tag = f'"'

        if start_tag in text and end_tag in text:
            return text.split(start_tag)[1].split(end_tag)[0].strip()
        return ""

    def _get_path_from(self, start_id: str, end_id: str) -> List[str]:
        """Get path nodes from start to end."""
        path = []
        current = end_id
        while current != start_id and current:
            path.append(current)
            current = self.nodes[current]["parent_id"]
        return path[::-1]  # Reverse to start->end

    def add_student_step(self, step_content: str, method_used: Optional[str] = None) -> str:
        """Add student step as a branch."""
        if not self.current_student_path:
            parent_id = self.root_node_id
        else:
            parent_id = self.current_student_path[-1]

        if method_used is None:
            method_used = self._detect_student_method(step_content)

        node_id = self.add_branch(parent_id, method_used, step_content, owner="student")
        self.current_student_path.append(node_id)
        return node_id

    def complete_student_path(self, success: bool, final_answer: Optional[str] = None):
        """Complete student path."""
        if self.current_student_path:
            leaf_id = self.current_student_path[-1]
            self.nodes[leaf_id]["type"] = "leaf"
            self.nodes[leaf_id]["success"] = success
            self.nodes[leaf_id]["final_answer"] = final_answer
        self.current_student_path = []  # Reset for next path

    def _detect_student_method(self, content: str) -> str:
        """Detect method from content (unchanged)."""
        content_lower = content.lower()
        if any(word in content_lower for word in ["equation", "solve for", "variable", "x ="]):
            return "algebraic"
        elif any(word in content_lower for word in ["diagram", "graph", "shape", "angle", "area"]):
            return "geometric"
        elif any(word in content_lower for word in ["calculate", "compute", "number", "digit"]):
            return "computational"
        elif any(word in content_lower for word in ["logic", "reason", "therefore", "because", "since", "if then"]):
            return "logical"
        elif any(word in content_lower for word in ["guess", "try", "maybe", "perhaps", "i think"]):
            return "trial_and_error"
        else:
            return "unknown"

    def _get_depth(self, node_id: str) -> int:
        """Calculate depth of a node."""
        depth = 0
        current = node_id
        while current != self.root_node_id and self.nodes[current]["parent_id"]:
            depth += 1
            current = self.nodes[current]["parent_id"]
        return depth

    def verify_paths_via_contradiction(self, model: Optional[Any] = None) -> Dict[str, str]:
        """Verify paths using proof by contradiction (simulate debate). Requires a model for prompting if not heuristic."""
        results = {}
        paths = self._get_all_paths()
        for i, path1 in enumerate(paths):
            for j, path2 in enumerate(paths):
                if i != j:
                    # Simple heuristic: Check if final answers conflict
                    ans1 = self.nodes[path1[-1]].get("final_answer", "")
                    ans2 = self.nodes[path2[-1]].get("final_answer", "")
                    if ans1 and ans2 and ans1 != ans2:
                        results[f"path_{i}_vs_{j}"] = "Contradiction detected: conflicting answers."
                        self.nodes[path1[-1]]["is_valid"] = False  # Mark one as invalid (arbitrary)
                    else:
                        # If model provided, prompt for deeper check
                        if model:
                            prompt = f"Assume {self.nodes[path1[1]]['content']}; does it contradict {self.nodes[path2[1]]['content']}?"
                            response = model.invoke([{"role": "user", "content": prompt}])  # Assuming model interface
                            if "contradiction" in response.lower():
                                self.nodes[path1[-1]]["is_valid"] = False
                                results[f"path_{i}_vs_{j}"] = response
        return results

    def prune_invalid_paths(self):
        """Prune invalid branches."""
        to_remove = [nid for nid, node in self.nodes.items() if not node["is_valid"] and nid != self.root_node_id]
        for nid in to_remove:
            parent = self.nodes[nid]["parent_id"]
            if parent:
                self.nodes[parent]["children"].remove(nid)
            del self.nodes[nid]

    def summarize_paths(self) -> str:
        """Summarize paths: consensus or ranked alternatives."""
        valid_paths = [p for p in self._get_all_paths() if self.nodes[p[-1]]["is_valid"]]
        answers = [self.nodes[p[-1]].get("final_answer", "") for p in valid_paths]
        if len(set(answers)) == 1:
            return f"Consensus: {answers[0]}"
        else:
            # Rank by average innovation
            ranked = sorted(valid_paths, key=lambda p: self._avg_weight(p, "innovation"), reverse=True)
            return f"Alternatives (ranked by innovation): {', '.join([self.nodes[p[-1]].get('final_answer', '') for p in ranked])}"

    def _get_all_paths(self) -> List[List[str]]:
        """Get all root-to-leaf paths."""
        paths = []

        def dfs(node_id: str, current_path: List[str]):
            current_path.append(node_id)
            if not self.nodes[node_id]["children"]:
                paths.append(current_path[:])
            for child in self.nodes[node_id]["children"]:
                dfs(child, current_path)
            current_path.pop()

        dfs(self.root_node_id, [])
        return paths

    def _avg_weight(self, path: List[str], weight_key: str) -> float:
        weights = [self.nodes[nid]["weight"][weight_key] for nid in path[1:]]  # Skip root
        return sum(weights) / len(weights) if weights else 0.0

    def set_gold_tree(self, gold_paths: List[Dict]):
        """Set gold standard paths."""
        self.gold_paths = gold_paths

    def compare_with_expert(self) -> Dict:
        """Compare student paths with gold (enhanced)."""
        student_paths = [p for p in self._get_all_paths() if self.nodes[p[1]].get("owner") == "student"]  # Updated to use "owner"
        if not student_paths or not self.gold_paths:
            return {"similarity": 0, "closest_expert_path": None}

        # Simple method overlap similarity
        max_sim = 0
        closest = None
        for gold in self.gold_paths:
            gold_methods = set([gold.get("method", '')])
            for sp in student_paths:
                student_methods = set([self.nodes[nid]["edge_method"] for nid in sp[1:]])
                sim = len(gold_methods & student_methods) / len(gold_methods | student_methods) if gold_methods else 0
                if sim > max_sim:
                    max_sim = sim
                    closest = gold
        return {"similarity": max_sim, "closest_expert_path": closest}

    def to_json(self) -> str:
        """Export to JSON."""
        data = {
            "problem_statement": self.problem_statement,
            "nodes": self.nodes,
            "gold_paths": self.gold_paths,
            "current_student_path": self.current_student_path
        }
        return json.dumps(data, indent=2)

    def from_json(self, json_str: str):
        """Import from JSON."""
        data = json.loads(json_str)
        self.problem_statement = data["problem_statement"]
        self.nodes = data["nodes"]
        self.gold_paths = data["gold_paths"]
        self.current_student_path = data["current_student_path"]
        # Update next_node_id based on existing nodes
        self.next_node_id = max([int(nid.split("_")[1]) for nid in self.nodes if "_" in nid] or [0]) + 1

    def visualize(self, owner=None) -> str:
        """Simple text-based visualization."""

        def build_tree(node_id: str, prefix: str = "") -> str:
            node = self.nodes[node_id]
            if owner and node['owner'] != owner and node_id != self.root_node_id:
                return []
            lines = \
                [f"{prefix}{node_id}: {node['content'][:50]}... (method: {node['edge_method']}, weights: {node['weight']}, owner: {node['owner']})"]
            children = [child for child in node["children"] if not owner or self.nodes[child]['owner'] == owner]
            for i, child in enumerate(children):
                child_prefix = prefix + ("└── " if i == len(children) - 1 else "├── ")
                lines.extend(build_tree(child, child_prefix))
            return lines

        return "\n".join(build_tree(self.root_node_id))

    def visualize_graph(self, filename='tree', owner=None):
        """Graphical visualization using graphviz with optional owner filter."""
        dot = graphviz.Digraph(format='png',
                               graph_attr={'rankdir': 'TB'},
                               node_attr={'shape': 'box', 'style': 'filled',
                                          'fontname': 'Arial', 'fontsize': '10'})

        # 计算复杂性权重范围
        complexities = []
        for node_id, node in self.nodes.items():
            if owner and node['owner'] != owner and node_id != self.root_node_id:
                continue
            complexities.append(node['weight']['complexity'])

        if complexities:
            min_complexity = min(complexities)
            max_complexity = max(complexities)
        else:
            min_complexity = 0
            max_complexity = 1

        # 颜色映射函数
        def get_color(complexity):
            if max_complexity == min_complexity:
                normalized = 0.5
            else:
                normalized = (complexity - min_complexity) / (max_complexity - min_complexity)
            intensity = int(255 * (1 - normalized * 0.7))
            return f"#{intensity:02x}{intensity:02x}ff"

        # 添加节点和边
        for node_id, node in self.nodes.items():
            if owner and node['owner'] != owner and node_id != self.root_node_id:
                continue

            complexity = node['weight']['complexity']
            color = get_color(complexity)
            label = f"{node_id}\\nContent: {node['content'][:20]}...\\nMethod: {node['edge_method']}\\nComplexity: {complexity:.2f}\\nOwner: {node['owner']}"
            dot.node(node_id, label, fillcolor=color)

        for node_id, node in self.nodes.items():
            if owner and node['owner'] != owner and node_id != self.root_node_id:
                continue
            for child in node["children"]:
                if not owner or self.nodes[child]['owner'] == owner:
                    dot.edge(node_id, child, label=node['edge_method'])

        # 在图的底部添加色阶说明
        color_legend = f"Color Legend: Complexity Weight ({min_complexity:.2f} = light blue, {max_complexity:.2f} = dark blue)"
        dot.attr(label=color_legend, labelloc='b', fontsize='12')

        dot.render(filename, cleanup=True)
        return f"Generated {filename}.png"

    # def visualize_graph(self, filename='tree', owner=None):
    #     """Graphical visualization using graphviz with optional owner filter.
    #
    #     Generates a PNG file. Requires graphviz installed.
    #     Colors represent node complexity weight with colorbar on the right.
    #     """
    #     # 创建主图和色阶条图
    #     dot = graphviz.Digraph(format='png',
    #                            graph_attr={'rankdir': 'TB'},
    #                            node_attr={'shape': 'box', 'style': 'filled',
    #                                       'fontname': 'Arial', 'fontsize': '10'})
    #
    #     # 计算复杂性权重的范围
    #     complexities = []
    #     for node_id, node in self.nodes.items():
    #         if owner and node['owner'] != owner and node_id != self.root_node_id:
    #             continue
    #         complexities.append(node['weight']['complexity'])
    #
    #     if complexities:
    #         min_complexity = min(complexities)
    #         max_complexity = max(complexities)
    #     else:
    #         min_complexity = 0
    #         max_complexity = 1
    #
    #     # 将复杂性权重映射到颜色（从浅蓝到深蓝）
    #     def get_color(complexity):
    #         # 归一化到0-1范围
    #         if max_complexity == min_complexity:
    #             normalized = 0.5
    #         else:
    #             normalized = (complexity - min_complexity) / (max_complexity - min_complexity)
    #
    #         # 使用蓝色系，复杂性越高颜色越深
    #         intensity = int(255 * (1 - normalized * 0.7))  # 保留一些基础亮度
    #         return f"#{intensity:02x}{intensity:02x}ff"
    #
    #     def add_node(node_id):
    #         node = self.nodes[node_id]
    #         if owner and node['owner'] != owner and node_id != self.root_node_id:
    #             return
    #
    #         # 根据复杂性设置节点颜色
    #         complexity = node['weight']['complexity']
    #         color = get_color(complexity)
    #
    #         label = f"{node_id}\\nContent: {node['content'][:20]}...\\nMethod: {node['edge_method']}\\nOwner: {node['owner']}"
    #         dot.node(node_id, label, fillcolor=color)
    #
    #         children = [child for child in node["children"] if not owner or self.nodes[child]['owner'] == owner]
    #         for child in children:
    #             dot.edge(node_id, child)
    #             add_node(child)
    #
    #     add_node(self.root_node_id)
    #
    #     # 创建带有色阶条的主图
    #     # self._add_colorbar_to_graph(dot, min_complexity, max_complexity, get_color)
    #     self._add_simple_colorbar(dot, min_complexity, max_complexity, get_color)
    #
    #     dot.render(filename, cleanup=True)
    #     return f"Generated {filename}.png"
    #
    # def _add_simple_colorbar(self, dot, min_val, max_val, color_func):
    #     """添加简单的色阶条"""
    #     # 创建色阶条HTML
    #     colorbar_height = 200
    #     colorbar_width = 30
    #
    #     # 构建渐变表格
    #     gradient_table = '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="0">'
    #     num_steps = 20
    #     for i in range(num_steps):
    #         value = min_val + (max_val - min_val) * (num_steps - 1 - i) / (num_steps - 1)  # 反转顺序
    #         color = color_func(value)
    #         gradient_table += f'<TR><TD BGCOLOR="{color}" WIDTH="{colorbar_width}" HEIGHT="{colorbar_height // num_steps}"></TD></TR>'
    #     gradient_table += '</TABLE>'
    #
    #     # 创建完整的色阶条标签
    #     html_label = f'''<
    # <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
    #   <TR>
    #     <TD>{gradient_table}</TD>
    #     <TD CELLPADDING="5" VALIGN="TOP"><FONT POINT-SIZE="10">{max_val:.2f}</FONT></TD>
    #   </TR>
    #   <TR>
    #     <TD></TD>
    #     <TD CELLPADDING="5" VALIGN="BOTTOM"><FONT POINT-SIZE="10">{min_val:.2f}</FONT></TD>
    #   </TR>
    #   <TR>
    #     <TD COLSPAN="2" ALIGN="CENTER"><FONT POINT-SIZE="10">Complexity</FONT></TD>
    #   </TR>
    # </TABLE>
    # >'''
    #
    #     # 添加色阶条节点
    #     dot.node('colorbar',
    #              label=html_label,
    #              shape='none',
    #              pos='e,100,100!')  # 尝试固定位置
    #
    #     # 将色阶条与根节点连接（不可见边，用于布局）
    #     dot.edge(self.root_node_id, 'colorbar', style='invis')

    def _add_colorbar_to_graph(self, dot, min_val, max_val, color_func):
        """在主图右侧添加色阶条 - 简化版本"""
        # 创建一个子图来放置色阶条
        with dot.subgraph(name='cluster_colorbar') as cb:
            cb.attr(rank='same', style='filled', color='lightgray',
                    label='Complexity Weight', fontsize='12',
                    labelloc='b', margin='15')

            # 创建垂直色阶条
            num_steps = 6
            for i in range(num_steps):
                value = min_val + (max_val - min_val) * i / (num_steps - 1)
                color = color_func(value)
                label = f"{value:.2f}"

                cb.node(f'cb_{i}', label=label,
                        fillcolor=color, width='0.8', height='0.3')

                if i > 0:
                    cb.edge(f'cb_{i - 1}', f'cb_{i}', style='invis')

            # 添加标题
            cb.node('cb_title', 'Low ← Complexity → High',
                    shape='plaintext', fontsize='10')
            cb.edge('cb_title', 'cb_0', style='invis')

    @property
    def solution_paths(self) -> List[Dict]:
        """Emulate old flat solution_paths for backward compatibility."""
        paths = []
        all_paths = self._get_all_paths()
        for i, path in enumerate(all_paths):
            path_dict = {
                "type": self.nodes[path[1]].get("owner", "unknown"),  # Use owner as type
                "path_id": i + 1,
                "method": self.nodes[path[1]]["edge_method"],
                "complexity": self._avg_weight(path, "complexity"),
                "innovation": self._avg_weight(path, "innovation"),
                "steps": [self.nodes[nid]["content"] for nid in path[1:]],
                "intermediate_answers": [self.nodes[nid].get("intermediate_answer", "") for nid in path[1:]],
                "final_answer": self.nodes[path[-1]].get("final_answer", "")
            }
            paths.append(path_dict)
        return paths



