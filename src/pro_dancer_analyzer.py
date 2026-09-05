"""
专业舞者影响分析模块
分析不同pro dancer对选手表现的影响
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


class ProDancerAnalyzer:
    """专业舞者分析器"""

    def __init__(self, all_results: List[Dict], raw_data: pd.DataFrame):
        self.results = all_results
        self.raw_data = raw_data

    def analyze_dancer_impact(self) -> Dict:
        """分析所有舞者的影响"""
        print("\n[专业舞者影响分析]")
        print("-" * 70)

        # 收集舞者数据
        dancer_stats = defaultdict(lambda: {
            'appearances': 0,
            'wins': 0,
            'top3': 0,
            'avg_placement': [],
            'avg_judge_score': [],
            'avg_fan_votes': [],
            'celebrities': []
        })

        # 从原始数据中提取
        for _, row in self.raw_data.iterrows():
            dancer = row['ballroom_partner']
            if pd.isna(dancer):
                continue

            celebrity = row['celebrity_name']
            placement = row['placement']

            dancer_stats[dancer]['appearances'] += 1
            dancer_stats[dancer]['celebrities'].append(celebrity)

            if not pd.isna(placement):
                dancer_stats[dancer]['avg_placement'].append(placement)
                if placement == 1:
                    dancer_stats[dancer]['wins'] += 1
                if placement <= 3:
                    dancer_stats[dancer]['top3'] += 1

        # 从推断结果中添加数据
        for result in self.results:
            for i, contestant_name in enumerate(result['contestants']):
                # 查找对应的舞者
                dancer = self._find_dancer_for_contestant(
                    contestant_name, result['season'], result['week']
                )

                if dancer:
                    judge_score = result['judge_scores'][i]
                    fan_vote = result['fan_vote_mean'][i]

                    dancer_stats[dancer]['avg_judge_score'].append(judge_score)
                    dancer_stats[dancer]['avg_fan_votes'].append(fan_vote)

        # 计算汇总统计
        summary = {}
        for dancer, stats in dancer_stats.items():
            if stats['appearances'] >= 3:  # 至少3次出场
                summary[dancer] = {
                    'appearances': stats['appearances'],
                    'wins': stats['wins'],
                    'top3': stats['top3'],
                    'win_rate': stats['wins'] / stats['appearances'],
                    'top3_rate': stats['top3'] / stats['appearances'],
                    'avg_placement': np.mean(stats['avg_placement']) if stats['avg_placement'] else np.nan,
                    'avg_judge_score': np.mean(stats['avg_judge_score']) if stats['avg_judge_score'] else np.nan,
                    'avg_fan_votes': np.mean(stats['avg_fan_votes']) if stats['avg_fan_votes'] else np.nan,
                    'celebrities': list(set(stats['celebrities']))
                }

        print(f"✓ 分析了 {len(summary)} 位专业舞者")

        return summary

    def _find_dancer_for_contestant(self,
                                    contestant: str,
                                    season: int,
                                    week: int) -> str:
        """查找选手对应的舞者"""
        match = self.raw_data[
            (self.raw_data['celebrity_name'] == contestant) &
            (self.raw_data['season'] == season)
            ]

        if not match.empty:
            return match.iloc[0]['ballroom_partner']
        return None

    def generate_dancer_ranking(self, dancer_stats: Dict) -> pd.DataFrame:
        """生成舞者排名表"""
        rows = []

        for dancer, stats in dancer_stats.items():
            rows.append({
                'Dancer': dancer,
                'Appearances': stats['appearances'],
                'Wins': stats['wins'],
                'Top 3': stats['top3'],
                'Win Rate': f"{stats['win_rate']:.1%}",
                'Top 3 Rate': f"{stats['top3_rate']:.1%}",
                'Avg Placement': f"{stats['avg_placement']:.1f}" if not np.isnan(stats['avg_placement']) else 'N/A',
                'Avg Judge Score': f"{stats['avg_judge_score']:.1f}" if not np.isnan(
                    stats['avg_judge_score']) else 'N/A',
                'Avg Fan Votes': f"{stats['avg_fan_votes'] / 1e6:.2f}M" if not np.isnan(
                    stats['avg_fan_votes']) else 'N/A'
            })

        df = pd.DataFrame(rows)

        # 按胜率排序
        df = df.sort_values('Win Rate', ascending=False)

        return df

    def analyze_dancer_celebrity_interaction(self,
                                             dancer_stats: Dict) -> Dict:
        """分析舞者与不同类型名人的配合效果"""
        print("\n分析舞者-名人类型交互效应...")

        interactions = {}

        for dancer, stats in dancer_stats.items():
            if stats['appearances'] < 5:
                continue

            # 统计与不同类型名人的合作
            celebrity_types = defaultdict(list)

            for celebrity in stats['celebrities']:
                celeb_data = self.raw_data[
                    self.raw_data['celebrity_name'] == celebrity
                    ]

                if not celeb_data.empty:
                    industry = celeb_data.iloc[0]['celebrity_industry']
                    placement = celeb_data.iloc[0]['placement']

                    if not pd.isna(industry) and not pd.isna(placement):
                        celebrity_types[industry].append(placement)

            # 计算每种类型的平均排名
            type_performance = {}
            for industry, placements in celebrity_types.items():
                type_performance[industry] = {
                    'count': len(placements),
                    'avg_placement': np.mean(placements)
                }

            interactions[dancer] = type_performance

        return interactions

    def plot_top_dancers(self,
                         dancer_stats: Dict,
                         top_n: int = 15,
                         save_path: str = None):
        """绘制顶级舞者对比图"""
        # 按胜率排序
        sorted_dancers = sorted(
            dancer_stats.items(),
            key=lambda x: (x[1]['win_rate'], x[1]['top3_rate']),
            reverse=True
        )[:top_n]

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))

        dancers = [d[0] for d in sorted_dancers]
        stats = [d[1] for d in sorted_dancers]

        # 子图1: 胜率和Top3率
        ax1 = axes[0, 0]
        x = np.arange(len(dancers))
        width = 0.35

        win_rates = [s['win_rate'] * 100 for s in stats]
        top3_rates = [s['top3_rate'] * 100 for s in stats]

        ax1.barh([i - width / 2 for i in x], win_rates, width,
                 label='Win Rate', color='#F18F01', alpha=0.8)
        ax1.barh([i + width / 2 for i in x], top3_rates, width,
                 label='Top 3 Rate', color='#06A77D', alpha=0.8)

        ax1.set_yticks(x)
        ax1.set_yticklabels(dancers, fontsize=10)
        ax1.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Success Rates', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')

        # 子图2: 出场次数 vs 胜场
        ax2 = axes[0, 1]
        appearances = [s['appearances'] for s in stats]
        wins = [s['wins'] for s in stats]

        scatter = ax2.scatter(appearances, wins, s=200, alpha=0.6,
                              c=win_rates, cmap='RdYlGn', edgecolors='black', linewidth=1.5)

        for i, dancer in enumerate(dancers):
            if wins[i] >= 2:  # 标注多次获胜的舞者
                ax2.annotate(dancer.split()[0], (appearances[i], wins[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=9)

        ax2.set_xlabel('Total Appearances', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total Wins', fontsize=12, fontweight='bold')
        ax2.set_title('Experience vs Success', fontsize=14, fontweight='bold')
        plt.colorbar(scatter, ax=ax2, label='Win Rate')
        plt.colorbar(scatter, ax=ax2, label='Win Rate (%)')
        ax2.grid(True, alpha=0.3)

        # 子图3: 平均排名
        ax3 = axes[1, 0]
        avg_placements = [s['avg_placement'] for s in stats if not np.isnan(s['avg_placement'])]
        dancers_with_placement = [dancers[i] for i, s in enumerate(stats) if not np.isnan(s['avg_placement'])]

        colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(avg_placements)))
        bars = ax3.barh(range(len(dancers_with_placement)), avg_placements,
                        color=colors, alpha=0.8, edgecolor='black')

        ax3.set_yticks(range(len(dancers_with_placement)))
        ax3.set_yticklabels(dancers_with_placement, fontsize=10)
        ax3.set_xlabel('Average Placement (Lower is Better)', fontsize=12, fontweight='bold')
        ax3.set_title('Average Final Placement', fontsize=14, fontweight='bold')
        ax3.axvline(x=5, color='red', linestyle='--', alpha=0.5, label='Top 5 Threshold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.invert_xaxis()

        # 子图4: 评委分 vs 粉丝票
        ax4 = axes[1, 1]
        judge_scores = [s['avg_judge_score'] for s in stats if not np.isnan(s['avg_judge_score'])]
        fan_votes = [s['avg_fan_votes'] / 1e6 for s in stats if not np.isnan(s['avg_fan_votes'])]
        dancers_with_both = [dancers[i] for i, s in enumerate(stats)
                             if not np.isnan(s['avg_judge_score']) and not np.isnan(s['avg_fan_votes'])]

        if len(judge_scores) > 0 and len(fan_votes) > 0:
            scatter2 = ax4.scatter(judge_scores, fan_votes, s=200, alpha=0.6,
                                   c=range(len(judge_scores)), cmap='viridis',
                                   edgecolors='black', linewidth=1.5)

            for i, dancer in enumerate(dancers_with_both[:5]):  # 标注前5名
                ax4.annotate(dancer.split()[0], (judge_scores[i], fan_votes[i]),
                             xytext=(5, 5), textcoords='offset points', fontsize=9)

            ax4.set_xlabel('Avg Judge Score', fontsize=12, fontweight='bold')
            ax4.set_ylabel('Avg Fan Votes (Millions)', fontsize=12, fontweight='bold')
            ax4.set_title('Judge Appeal vs Fan Appeal', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)

        plt.suptitle(f'Top {top_n} Professional Dancers Analysis',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_dancer_celebrity_heatmap(self,
                                      interactions: Dict,
                                      save_path: str = None):
        """绘制舞者-名人类型交互热图"""
        # 准备数据
        dancers = []
        industries = set()

        for dancer, types in interactions.items():
            if len(types) >= 2:  # 至少与2种类型合作过
                dancers.append(dancer)
                industries.update(types.keys())

        industries = sorted(list(industries))

        # 创建矩阵
        matrix = np.full((len(dancers), len(industries)), np.nan)

        for i, dancer in enumerate(dancers):
            for j, industry in enumerate(industries):
                if industry in interactions[dancer]:
                    matrix[i, j] = interactions[dancer][industry]['avg_placement']

        # 绘图
        fig, ax = plt.subplots(figsize=(14, max(8, len(dancers) * 0.4)))

        # 使用反转的colormap（低排名=好=绿色）
        im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto', vmin=1, vmax=10)

        ax.set_xticks(range(len(industries)))
        ax.set_yticks(range(len(dancers)))
        ax.set_xticklabels(industries, rotation=45, ha='right')
        ax.set_yticklabels(dancers)

        # 添加数值标签
        for i in range(len(dancers)):
            for j in range(len(industries)):
                if not np.isnan(matrix[i, j]):
                    text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                                   ha="center", va="center", color="black", fontsize=9)

        ax.set_title('Dancer Performance by Celebrity Type\n(Average Placement - Lower is Better)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Celebrity Industry', fontsize=12, fontweight='bold')
        ax.set_ylabel('Professional Dancer', fontsize=12, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Avg Placement')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()