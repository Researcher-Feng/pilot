# visualization_tools.py
"""
实验结果可视化工具
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any


class ResultVisualizer:
    """实验结果可视化类"""

    def __init__(self, results_file: str):
        self.results_file = results_file
        self.data = self.load_results()

    def load_results(self) -> Dict[str, Any]:
        """加载结果文件"""
        with open(self.results_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_comprehensive_dashboard(self, output_file: str = "comprehensive_dashboard.png"):
        """创建综合仪表板"""
        fig = plt.figure(figsize=(20, 16))

        # 创建子图网格
        gs = fig.add_gridspec(4, 4)

        # 1. 主要指标（左上）
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_main_metrics(ax1)

        # 2. 对话轮次分布（右上）
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_turn_distribution(ax2)

        # 3. 教师意图分析（中左）
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_teacher_intents(ax3)

        # 4. 思考模式分析（中右）
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_thinking_patterns(ax4)

        # 5. 准确率随时间变化（下左）
        ax5 = fig.add_subplot(gs[2, :2])
        self._plot_accuracy_progression(ax5)

        # 6. 答案泄露分析（下右）
        ax6 = fig.add_subplot(gs[2, 2:])
        self._plot_answer_leakage(ax6)

        # 7. 详细统计表格（底部）
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_statistics_table(ax7)

        plt.suptitle('Multi-Agent Math Tutoring System - Comprehensive Analysis',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()

        return output_file

    def _plot_main_metrics(self, ax):
        """绘制主要指标"""
        summary = self.data.get('summary', {})

        metrics = ['Accuracy', 'Avg Turns', 'Parallel Thinking', 'Thinking Paths']
        values = [
            summary.get('accuracy', 0),
            summary.get('avg_turns_per_problem', 0),
            summary.get('avg_parallel_thinking', 0),
            summary.get('avg_thinking_paths', 0)
        ]
        colors = ['#2ecc71', '#3498db', '#9b59b6', '#f39c12']

        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_title('Key Performance Metrics', fontweight='bold')
        ax.set_ylabel('Value')

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    def _plot_turn_distribution(self, ax):
        """绘制对话轮次分布"""
        records = self.data.get('records', [])
        turn_counts = [r.get('total_turns', 0) for r in records]

        if turn_counts:
            ax.hist(turn_counts, bins=range(1, max(turn_counts) + 2),
                    alpha=0.7, color='#3498db', edgecolor='black')
            ax.set_title('Distribution of Dialogue Turns', fontweight='bold')
            ax.set_xlabel('Number of Turns')
            ax.set_ylabel('Frequency')

            # 添加平均线
            avg_turns = np.mean(turn_counts)
            ax.axvline(avg_turns, color='red', linestyle='--',
                       label=f'Average: {avg_turns:.2f}')
            ax.legend()

    def _plot_teacher_intents(self, ax):
        """绘制教师意图分析"""
        records = self.data.get('records', [])
        intents = []

        for record in records:
            for turn in record.get('turns', []):
                intent = turn.get('teacher_intent', '')
                if intent:
                    intents.append(intent)

        if intents:
            intent_counts = pd.Series(intents).value_counts()

            # 使用饼图展示
            wedges, texts, autotexts = ax.pie(
                intent_counts.values,
                labels=intent_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=plt.cm.Set3(np.linspace(0, 1, len(intent_counts)))
            )

            ax.set_title('Teacher Response Intent Distribution', fontweight='bold')

            # 美化文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

    def _plot_thinking_patterns(self, ax):
        """绘制思考模式分析"""
        records = self.data.get('records', [])

        parallel_counts = [r.get('parallel_thinking_count', 0) for r in records]
        path_counts = [r.get('thinking_paths_count', 0) for r in records]

        x = range(len(records))
        width = 0.35

        ax.bar([i - width / 2 for i in x], parallel_counts, width,
               label='Parallel Thinking', color='#9b59b6', alpha=0.7)
        ax.bar([i + width / 2 for i in x], path_counts, width,
               label='Thinking Paths', color='#f39c12', alpha=0.7)

        ax.set_title('Thinking Patterns Across Problems', fontweight='bold')
        ax.set_xlabel('Problem Index')
        ax.set_ylabel('Count')
        ax.legend()

        # 只显示部分x轴标签以避免拥挤
        if len(records) > 10:
            ax.set_xticks(range(0, len(records), max(1, len(records) // 10)))

    def _plot_accuracy_progression(self, ax):
        """绘制准确率随时间变化"""
        records = self.data.get('records', [])

        if records:
            # 计算累积准确率
            correct_count = 0
            cumulative_accuracy = []

            for i, record in enumerate(records):
                if record.get('correct', False):
                    correct_count += 1
                cumulative_accuracy.append(correct_count / (i + 1))

            ax.plot(range(1, len(records) + 1), cumulative_accuracy,
                    marker='o', linewidth=2, markersize=4, color='#2ecc71')
            ax.set_title('Cumulative Accuracy Progression', fontweight='bold')
            ax.set_xlabel('Problem Number')
            ax.set_ylabel('Cumulative Accuracy')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)

    def _plot_answer_leakage(self, ax):
        """绘制答案泄露分析"""
        records = self.data.get('records', [])

        leaked = sum(1 for r in records if r.get('leaked_answer', False))
        not_leaked = len(records) - leaked

        categories = ['Leaked Answers', 'No Leakage']
        counts = [leaked, not_leaked]
        colors = ['#e74c3c', '#2ecc71']

        bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
        ax.set_title('Answer Leakage Analysis', fontweight='bold')
        ax.set_ylabel('Count')

        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')

    def _plot_statistics_table(self, ax):
        """绘制统计表格"""
        summary = self.data.get('summary', {})

        # 隐藏坐标轴
        ax.axis('tight')
        ax.axis('off')

        # 准备表格数据
        table_data = [
            ['Total Problems', summary.get('total_problems', 0)],
            ['Correct Answers', summary.get('correct_answers', 0)],
            ['Accuracy', f"{summary.get('accuracy', 0):.4f}"],
            ['Leaked Answers', summary.get('leaked_answers', 0)],
            ['Leakage Rate', f"{summary.get('answer_leakage_rate', 0):.4f}"],
            ['Avg Turns/Problem', f"{summary.get('avg_turns_per_problem', 0):.2f}"],
            ['Avg Parallel Thinking', f"{summary.get('avg_parallel_thinking', 0):.2f}"],
            ['Avg Thinking Paths', f"{summary.get('avg_thinking_paths', 0):.2f}"]
        ]

        table = ax.table(
            cellText=table_data,
            colLabels=['Metric', 'Value'],
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )

        # 美化表格
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        # 设置标题行样式
        for i in range(2):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')


def create_comparison_visualization(result_files: List[str], labels: List[str],
                                    output_file: str = "comparison_analysis.png"):
    """创建多实验对比可视化"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    all_data = []
    for file in result_files:
        with open(file, 'r', encoding='utf-8') as f:
            all_data.append(json.load(f))

    # 1. 准确率对比
    accuracies = [data.get('summary', {}).get('accuracy', 0) for data in all_data]
    bars1 = axes[0, 0].bar(labels, accuracies, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Accuracy Comparison', fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy')
    for bar, acc in zip(bars1, accuracies):
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

    # 2. 对话轮次对比
    avg_turns = [data.get('summary', {}).get('avg_turns_per_problem', 0) for data in all_data]
    bars2 = axes[0, 1].bar(labels, avg_turns, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Average Turns Comparison', fontweight='bold')
    axes[0, 1].set_ylabel('Turns per Problem')
    for bar, turns in zip(bars2, avg_turns):
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f'{turns:.2f}', ha='center', va='bottom', fontweight='bold')

    # 3. 答案泄露率对比
    leakage_rates = [data.get('summary', {}).get('answer_leakage_rate', 0) for data in all_data]
    bars3 = axes[0, 2].bar(labels, leakage_rates, color='lightcoral', alpha=0.7)
    axes[0, 2].set_title('Answer Leakage Rate Comparison', fontweight='bold')
    axes[0, 2].set_ylabel('Leakage Rate')
    for bar, rate in zip(bars3, leakage_rates):
        axes[0, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')

    # 4. 并行思考对比
    parallel_thinking = [data.get('summary', {}).get('avg_parallel_thinking', 0) for data in all_data]
    thinking_paths = [data.get('summary', {}).get('avg_thinking_paths', 0) for data in all_data]

    x = np.arange(len(labels))
    width = 0.35

    bars4a = axes[1, 0].bar(x - width / 2, parallel_thinking, width,
                            label='Parallel Thinking', alpha=0.7)
    bars4b = axes[1, 0].bar(x + width / 2, thinking_paths, width,
                            label='Thinking Paths', alpha=0.7)
    axes[1, 0].set_title('Thinking Patterns Comparison', fontweight='bold')
    axes[1, 0].set_ylabel('Average Count')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(labels)
    axes[1, 0].legend()

    # 5. 教师意图分布对比（简化版）
    intent_data = []
    for data in all_data:
        intents = []
        for record in data.get('records', []):
            for turn in record.get('turns', []):
                intent = turn.get('teacher_intent', '')
                if intent:
                    intents.append(intent)
        intent_counts = pd.Series(intents).value_counts()
        intent_data.append(intent_counts)

    # 选择前3种最常见的意图进行对比
    common_intents = set()
    for counts in intent_data:
        common_intents.update(counts.head(3).index)

    for i, intent in enumerate(common_intents):
        intent_values = [counts.get(intent, 0) for counts in intent_data]
        axes[1, 1].bar([f"{label}\n{intent}" for label in labels], intent_values,
                       alpha=0.7, label=intent)

    axes[1, 1].set_title('Common Teacher Intents Comparison', fontweight='bold')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # 6. 综合评分（自定义评分公式）
    scores = []
    for data in all_data:
        summary = data.get('summary', {})
        # 评分公式：准确率权重最高，泄露率负权重，思考模式正权重
        score = (summary.get('accuracy', 0) * 0.5 +
                 (1 - summary.get('answer_leakage_rate', 0)) * 0.3 +
                 min(summary.get('avg_parallel_thinking', 0) * 0.1, 0.1) +
                 min(summary.get('avg_thinking_paths', 0) * 0.1, 0.1))
        scores.append(score)

    bars6 = axes[1, 2].bar(labels, scores, color='gold', alpha=0.7)
    axes[1, 2].set_title('Overall Performance Score', fontweight='bold')
    axes[1, 2].set_ylabel('Score')
    for bar, score in zip(bars6, scores):
        axes[1, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.suptitle('Multi-Experiment Comparison Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    return output_file


if __name__ == '__main__':
    # 使用示例
    print("📊 实验结果可视化工具")
    print("使用方法:")
    print("1. 创建单个实验可视化:")
    print("   visualizer = ResultVisualizer('results/your_experiment.json')")
    print("   visualizer.create_comprehensive_dashboard()")
    print()
    print("2. 创建多实验对比:")
    print("   create_comparison_visualization(['exp1.json', 'exp2.json'], ['Exp1', 'Exp2'])")