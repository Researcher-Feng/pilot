
def visualize_graph(self, filename='tree', owner=None):
    """Graphical visualization using graphviz with enhanced styling.

    Features:
    - Different colors for different methods (with legend)
    - Gradient colors based on complexity
    - Cleaner node labels showing only ID, owner and content
    - No edge labels (methods shown in legend instead)

    Generates a PNG file. Requires graphviz installed.
    """
    # Define color scheme for different methods
    method_colors = {
        "algebraic": "#FF6B6B",      # Red
        "geometric": "#4ECDC4",      # Teal
        "logical": "#45B7D1",        # Blue
        "computational": "#96CEB4",  # Green
        "equations": "#FFEAA7",      # Yellow
        "disproof": "#DDA0DD",       # Plum
        "trial_and_error": "#FFA07A", # Light Salmon
        "unknown": "#B0B0B0"         # Gray
    }

    # Create main graph
    dot = graphviz.Digraph(
        format='png',
        graph_attr={
            'rankdir': 'TB',
            'bgcolor': 'white',
            'fontname': 'Arial',
            'fontsize': '12'
        },
        node_attr={
            'shape': 'box',
            'style': 'filled,rounded',
            'filled': 'true',
            'fontname': 'Arial',
            'fontsize': '10',
            'height': '0.6',
            'width': '1.2'
        },
        edge_attr={
            'color': '#555555',
            'arrowsize': '0.8',
            'penwidth': '1.5'
        }
    )

    # Add title
    dot.attr(label=f'Solution Tree: {self.problem_statement[:50]}...\n\n',
             labelloc='t', labeljust='c')

    # Method legend subgraph (align to left)
    with dot.subgraph(name='cluster_legend_methods') as legend_methods:
        legend_methods.attr(
            label='Methods (by color)',
            style='filled,rounded',
            color='lightgray',
            fontname='Arial',
            fontsize='10'
        )

        # Add method color items
        for i, (method, color) in enumerate(method_colors.items()):
            legend_methods.node(
                f'legend_method_{method}',
                label=method,
                shape='box',
                style='filled',
                fillcolor=color,
                fontname='Arial',
                fontsize='9',
                width='1.5',
                height='0.4'
            )

    # Complexity gradient legend subgraph (align to right)
    with dot.subgraph(name='cluster_legend_complexity') as legend_complexity:
        legend_complexity.attr(
            label='Complexity (darker = higher)',
            style='filled,rounded',
            color='lightgray',
            fontname='Arial',
            fontsize='10'
        )

        # Create gradient samples
        complexities = [0.0, 0.25, 0.5, 0.75, 1.0]
        for i, comp in enumerate(complexities):
            # Calculate gradient color (light to dark blue)
            intensity = int(255 * (1 - comp * 0.7))  # Keep some brightness
            color = f'#0000{intensity:02X}'

            legend_complexity.node(
                f'legend_comp_{i}',
                label=f'{comp:.2f}',
                shape='box',
                style='filled',
                fillcolor=color,
                fontcolor='white' if comp > 0.5 else 'black',
                fontname='Arial',
                fontsize='9',
                width='0.8',
                height='0.4'
            )

    # Add all nodes to main graph
    def add_node(node_id):
        node = self.nodes[node_id]

        # Apply owner filter
        if owner and node['owner'] != owner and node_id != self.root_node_id:
            return

        # Get method and complexity for styling
        method = node['edge_method'] or 'unknown'
        complexity = node['weight']['complexity']

        # Determine base color based on method
        base_color = method_colors.get(method, method_colors['unknown'])

        # Apply complexity gradient (darken based on complexity)
        if complexity > 0:
            # Convert hex to RGB and darken
            r, g, b = int(base_color[1:3], 16), int(base_color[3:5], 16), int(base_color[5:7], 16)
            darken_factor = 0.3 * complexity  # Darken up to 30%
            r = int(r * (1 - darken_factor))
            g = int(g * (1 - darken_factor))
            b = int(b * (1 - darken_factor))
            node_color = f'#{r:02X}{g:02X}{b:02X}'
        else:
            node_color = base_color

        # Create clean label
        content_preview = node['content'][:30] + '...' if len(node['content']) > 30 else node['content']
        label = f"{node_id}\\nOwner: {node['owner'] or 'none'}\\n{content_preview}"

        # Special styling for root node
        if node_id == self.root_node_id:
            dot.node(
                node_id,
                label=f"ROOT: {content_preview}",
                style='filled,rounded,bold',
                fillcolor='#FFD700',  # Gold color for root
                fontcolor='black',
                fontsize='11'
            )
        else:
            dot.node(
                node_id,
                label=label,
                fillcolor=node_color,
                fontcolor='white' if complexity > 0.5 else 'black',
                gradientangle='90'
            )

        # Add edges to children (filtered by owner)
        children = [child for child in node["children"]
                    if not owner or self.nodes[child]['owner'] == owner]

        for child in children:
            # Don't add method label to edge - shown in legend instead
            dot.edge(node_id, child)
            add_node(child)

    # Start building from root
    add_node(self.root_node_id)

    # Render the graph
    dot.render(filename, cleanup=True)
    return f"Generated {filename}.png"

