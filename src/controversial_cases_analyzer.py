"""
争议案例分析器
专门分析Jerry Rice, Billy Ray Cyrus, Bristol Palin, Bobby Bones等争议选手
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class ControversialCasesAnalyzer:
    """争议案例分析器"""

    def __init__(self, all_results: List[Dict]):
        self.results = all_results

    def analyze_all_cases(self) -> Dict:
        """分析所有争议案例（包含Bobby Bones）"""
        controversial_celebrities = {
            'Jerry Rice': 2,
            'Billy Ray Cyrus': 4,
            'Bristol Palin': 11,
            'Bobby Bones': 27
        }

        analyses = {}

        for name, season in controversial_celebrities.items():
            print(f"\n分析争议案例: {name} (Season {season})")

            # 查找该选手的所有周数据
            contestant_weeks = [
                r for r in self.results
                if r['season'] == season and name in r['contestants']
            ]

            if len(contestant_weeks) == 0:
                print(f"  ⚠ 数据不足（需要扩展到Season {season}）")
                analyses[name] = None
                continue

            # 执行分析
            analysis = self._analyze_contestant(name, contestant_weeks)
            analyses[name] = analysis

            print(f"  ✓ 完成: {len(contestant_weeks)} 周数据")

        return analyses

    def _analyze_contestant(self, name: str, weeks: List[Dict]) -> Dict:
        """分析单个选手的争议程度"""

        # 收集数据
        judge_ranks = []
        fan_ranks = []
        judge_scores = []
        fan_votes = []

        for week in weeks:
            idx = week['contestants'].index(name)

            # 评委排名
            j_scores = week['judge_scores']
            j_rank = self._get_rank(j_scores, idx)
            judge_ranks.append(j_rank)
            judge_scores.append(j_scores[idx])

            # 粉丝排名
            f_votes = week['fan_vote_mean']
            f_rank = self._get_rank(f_votes, idx, reverse=True)
            fan_ranks.append(f_rank)
            fan_votes.append(f_votes[idx])

        # 计算统计量
        judge_stats = {
            'avg_rank': np.mean(judge_ranks),
            'best_rank': int(np.min(judge_ranks)),
            'worst_rank': int(np.max(judge_ranks)),
            'times_last': sum(1 for r in judge_ranks if r == max(judge_ranks)),
            'avg_score': np.mean(judge_scores)
        }

        fan_stats = {
            'avg_rank': np.mean(fan_ranks),
            'best_rank': int(np.min(fan_ranks)),
            'worst_rank': int(np.max(fan_ranks)),
            'times_first': sum(1 for r in fan_ranks if r == 1),
            'avg_votes': np.mean(fan_votes)
        }

        # 争议分数（粉丝排名 - 评委排名，正值表示粉丝更喜欢）
        controversy_score = judge_stats['avg_rank'] - fan_stats['avg_rank']

        # 系统影响分析
        system_outcomes = self._analyze_system_impact(name, weeks)

        return {
            'name': name,
            'season': weeks[0]['season'],
            'weeks_competed': len(weeks),
            'final_placement': self._get_final_placement(name, weeks),
            'judge_stats': judge_stats,
            'fan_stats': fan_stats,
            'controversy_score': controversy_score,
            'weekly_data': {
                'judge_ranks': judge_ranks,
                'fan_ranks': fan_ranks,
                'judge_scores': judge_scores,
                'fan_votes': fan_votes
            },
            'system_outcomes': system_outcomes
        }

    def _get_rank(self, values: np.ndarray, idx: int, reverse: bool = False) -> int:
        """获取排名（1=最好）"""
        if reverse:
            # 对于粉丝票数，高=好
            sorted_indices = np.argsort(values)[::-1]
        else:
            # 对于评委分数，高=好
            sorted_indices = np.argsort(values)[::-1]

        rank = np.where(sorted_indices == idx)[0][0] + 1
        return int(rank)

    def _get_final_placement(self, name: str, weeks: List[Dict]) -> int:
        """获取最终排名"""
        # 最后一周的排名
        last_week = weeks[-1]
        idx = last_week['contestants'].index(name)

        # 如果是被淘汰的，返回参赛人数
        if last_week['eliminated'] == name:
            return len(last_week['contestants'])

        # 否则返回基于总分的排名
        return self._get_rank(last_week['fan_vote_mean'], idx, reverse=True)

    def _analyze_system_impact(self, name: str, weeks: List[Dict]) -> Dict:
        """分析不同投票系统对该选手的影响"""

        rank_eliminations = []
        percentage_eliminations = []

        for week in weeks:
            idx = week['contestants'].index(name)
            j_scores = week['judge_scores']
            f_votes = week['fan_vote_mean']

            # Rank系统
            j_ranks = self._to_ranks(j_scores)
            f_ranks = self._to_ranks(f_votes)
            combined_rank = j_ranks + f_ranks
            would_eliminate_rank = (np.argmax(combined_rank) == idx)
            rank_eliminations.append(would_eliminate_rank)

            # Percentage系统
            j_pct = j_scores / j_scores.sum()
            f_pct = f_votes / f_votes.sum()
            combined_pct = j_pct + f_pct
            would_eliminate_pct = (np.argmin(combined_pct) == idx)
            percentage_eliminations.append(would_eliminate_pct)

        return {
            'rank_system': rank_eliminations,
            'percentage_system': percentage_eliminations,
            'rank_survival_rate': 1 - np.mean(rank_eliminations),
            'percentage_survival_rate': 1 - np.mean(percentage_eliminations)
        }

    @staticmethod
    def _to_ranks(values: np.ndarray) -> np.ndarray:
        """转换为排名"""
        order = np.argsort(values)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(values)) + 1
        return ranks

    def generate_comparison_report(self, analyses: Dict) -> pd.DataFrame:
        """生成争议案例对比报告"""

        rows = []
        for name, analysis in analyses.items():
            if analysis is None:
                rows.append({
                    'Celebrity': name,
                    'Season': 'N/A',
                    'Weeks': 0,
                    'Final Place': 'N/A',
                    'Avg Judge Rank': 'N/A',
                    'Avg Fan Rank': 'N/A',
                    'Controversy Score': 'N/A',
                    'Times Last (Judge)': 'N/A',
                    'Times First (Fan)': 'N/A'
                })
            else:
                rows.append({
                    'Celebrity': name,
                    'Season': analysis['season'],
                    'Weeks': analysis['weeks_competed'],
                    'Final Place': analysis['final_placement'],
                    'Avg Judge Rank': f"{analysis['judge_stats']['avg_rank']:.1f}",
                    'Avg Fan Rank': f"{analysis['fan_stats']['avg_rank']:.1f}",
                    'Controversy Score': f"{analysis['controversy_score']:.2f}",
                    'Times Last (Judge)': analysis['judge_stats']['times_last'],
                    'Times First (Fan)': analysis['fan_stats']['times_first']
                })

        return pd.DataFrame(rows)

    def plot_controversial_case(self,
                                name: str,
                                analysis: Dict,
                                save_path: str = None):
        """绘制单个争议案例的详细图表"""

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        weeks = list(range(1, analysis['weeks_competed'] + 1))
        judge_ranks = analysis['weekly_data']['judge_ranks']
        fan_ranks = analysis['weekly_data']['fan_ranks']
        judge_scores = analysis['weekly_data']['judge_scores']
        fan_votes = np.array(analysis['weekly_data']['fan_votes']) / 1e6

        # 子图1: 排名对比
        ax1 = axes[0, 0]
        ax1.plot(weeks, judge_ranks, 'o-', color='#D62246', linewidth=2,
                markersize=8, label='Judge Rank')
        ax1.plot(weeks, fan_ranks, 's-', color='#06A77D', linewidth=2,
                markersize=8, label='Fan Rank')
        ax1.axhline(y=1, color='gold', linestyle='--', linewidth=2,
                   alpha=0.5, label='1st Place')
        ax1.invert_yaxis()
        ax1.set_xlabel('Week', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Rank (1=Best)', fontsize=12, fontweight='bold')
        ax1.set_title(f'{name} - Weekly Rankings\nControversy Score: {analysis["controversy_score"]:.2f}',
                     fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 子图2: 分数/票数趋势
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()

        line1 = ax2.plot(weeks, judge_scores, 'o-', color='#D62246',
                        linewidth=2, markersize=8, label='Judge Score')
        line2 = ax2_twin.plot(weeks, fan_votes, 's-', color='#06A77D',
                             linewidth=2, markersize=8, label='Fan Votes (M)')

        ax2.set_xlabel('Week', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Judge Score', fontsize=12, fontweight='bold', color='#D62246')
        ax2_twin.set_ylabel('Fan Votes (Millions)', fontsize=12, fontweight='bold', color='#06A77D')
        ax2.set_title(f'{name} - Performance Metrics', fontsize=13, fontweight='bold')

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left')
        ax2.grid(True, alpha=0.3)

        # 子图3: 排名分布
        ax3 = axes[1, 0]
        x = np.arange(2)
        width = 0.35

        judge_avg = analysis['judge_stats']['avg_rank']
        fan_avg = analysis['fan_stats']['avg_rank']

        bars = ax3.bar(x, [judge_avg, fan_avg], width,
                      color=['#D62246', '#06A77D'], alpha=0.7, edgecolor='black')

        ax3.set_ylabel('Average Rank', fontsize=12, fontweight='bold')
        ax3.set_title(f'{name} - Average Rankings Comparison', fontsize=13, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Judges', 'Fans'])
        ax3.invert_yaxis()
        ax3.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='top',
                    fontsize=11, fontweight='bold')

        # 子图4: 系统影响
        ax4 = axes[1, 1]

        systems = ['Rank\nSystem', 'Percentage\nSystem']
        survival_rates = [
            analysis['system_outcomes']['rank_survival_rate'] * 100,
            analysis['system_outcomes']['percentage_survival_rate'] * 100
        ]

        bars = ax4.barh(systems, survival_rates, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax4.axvline(x=50, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax4.set_xlabel('Survival Rate (%)', fontsize=12, fontweight='bold')
        ax4.set_title(f'{name} - Voting System Impact', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='x')

        # 添加数值标签
        for i, (bar, rate) in enumerate(zip(bars, survival_rates)):
            ax4.text(rate, i, f' {rate:.1f}%', va='center',
                    fontsize=11, fontweight='bold')

        plt.suptitle(f'Controversial Case Analysis: {name} (Season {analysis["season"]})\n'
                    f'Final Placement: {analysis["final_placement"]} | '
                    f'Weeks Competed: {analysis["weeks_competed"]}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()