"""
特征影响深度分析模块
分离分析特征对judge分数和fan投票的不同影响
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class FeatureImpactAnalyzer:
    """特征影响分析器"""

    def __init__(self, all_results: List[Dict], raw_data: pd.DataFrame):
        self.results = all_results
        self.raw_data = raw_data

    def analyze_feature_impact(self) -> Dict:
        """分析特征对judge和fan的不同影响"""
        print("\n[特征影响深度分析]")
        print("-" * 70)

        # 收集数据
        data_points = []

        for result in self.results:
            season = result['season']
            week = result['week']

            for i, contestant in enumerate(result['contestants']):
                # 查找选手信息
                contestant_info = self._get_contestant_info(contestant, season)

                if contestant_info is not None:
                    data_points.append({
                        'contestant': contestant,
                        'season': season,
                        'week': week,
                        'judge_score': result['judge_scores'][i],
                        'fan_vote': result['fan_vote_mean'][i],
                        'age': contestant_info['age'],
                        'is_athlete': contestant_info['is_athlete'],
                        'is_singer': contestant_info['is_singer'],
                        'is_actor': contestant_info['is_actor'],
                        'dancer': contestant_info['dancer']
                    })

        df = pd.DataFrame(data_points)

        print(f"✓ 收集了 {len(df)} 个数据点")

        # 分别分析对judge和fan的影响
        judge_analysis = self._analyze_impact_on_judges(df)
        fan_analysis = self._analyze_impact_on_fans(df)

        # 对比分析
        comparison = self._compare_impacts(judge_analysis, fan_analysis)

        return {
            'judge_impact': judge_analysis,
            'fan_impact': fan_analysis,
            'comparison': comparison,
            'data': df
        }

    def _get_contestant_info(self, name: str, season: int) -> Dict:
        """获取选手信息"""
        match = self.raw_data[
            (self.raw_data['celebrity_name'] == name) &
            (self.raw_data['season'] == season)
        ]

        if match.empty:
            return None

        row = match.iloc[0]

        industry = str(row['celebrity_industry']).lower()

        return {
            'age': row['celebrity_age_during_season'],
            'is_athlete': 'athlete' in industry or 'sport' in industry,
            'is_singer': 'singer' in industry or 'music' in industry,
            'is_actor': 'actor' in industry or 'actress' in industry,
            'dancer': row['ballroom_partner']
        }

    def _analyze_impact_on_judges(self, df: pd.DataFrame) -> Dict:
        """分析特征对评委分数的影响"""
        results = {}

        # 年龄
        try:
            age_corr = stats.pearsonr(df['age'], df['judge_score'])
            results['age'] = {
                'correlation': age_corr[0],
                'p_value': age_corr[1],
                'significant': age_corr[1] < 0.05
            }
        except:
            results['age'] = {
                'correlation': 0.0,
                'p_value': 1.0,
                'significant': False
            }

        # 运动员
        try:
            athlete_scores = df[df['is_athlete']]['judge_score']
            non_athlete_scores = df[~df['is_athlete']]['judge_score']
            if len(athlete_scores) > 0 and len(non_athlete_scores) > 0:
                athlete_test = stats.ttest_ind(athlete_scores, non_athlete_scores)
                results['athlete'] = {
                    'mean_diff': athlete_scores.mean() - non_athlete_scores.mean(),
                    'p_value': athlete_test.pvalue,
                    'significant': athlete_test.pvalue < 0.05
                }
            else:
                results['athlete'] = {
                    'mean_diff': 0.0,
                    'p_value': 1.0,
                    'significant': False
                }
        except:
            results['athlete'] = {
                'mean_diff': 0.0,
                'p_value': 1.0,
                'significant': False
            }

        # 歌手
        try:
            singer_scores = df[df['is_singer']]['judge_score']
            non_singer_scores = df[~df['is_singer']]['judge_score']
            if len(singer_scores) > 0 and len(non_singer_scores) > 0:
                singer_test = stats.ttest_ind(singer_scores, non_singer_scores)
                results['singer'] = {
                    'mean_diff': singer_scores.mean() - non_singer_scores.mean(),
                    'p_value': singer_test.pvalue,
                    'significant': singer_test.pvalue < 0.05
                }
            else:
                results['singer'] = {
                    'mean_diff': 0.0,
                    'p_value': 1.0,
                    'significant': False
                }
        except:
            results['singer'] = {
                'mean_diff': 0.0,
                'p_value': 1.0,
                'significant': False
            }

        # 演员
        try:
            actor_scores = df[df['is_actor']]['judge_score']
            non_actor_scores = df[~df['is_actor']]['judge_score']
            if len(actor_scores) > 0 and len(non_actor_scores) > 0:
                actor_test = stats.ttest_ind(actor_scores, non_actor_scores)
                results['actor'] = {
                    'mean_diff': actor_scores.mean() - non_actor_scores.mean(),
                    'p_value': actor_test.pvalue,
                    'significant': actor_test.pvalue < 0.05
                }
            else:
                results['actor'] = {
                    'mean_diff': 0.0,
                    'p_value': 1.0,
                    'significant': False
                }
        except:
            results['actor'] = {
                'mean_diff': 0.0,
                'p_value': 1.0,
                'significant': False
            }

        return results

    def _analyze_impact_on_fans(self, df: pd.DataFrame) -> Dict:
        """分析特征对粉丝投票的影响"""
        results = {}

        # 年龄
        try:
            age_corr = stats.pearsonr(df['age'], df['fan_vote'])
            results['age'] = {
                'correlation': age_corr[0],
                'p_value': age_corr[1],
                'significant': age_corr[1] < 0.05
            }
        except:
            results['age'] = {
                'correlation': 0.0,
                'p_value': 1.0,
                'significant': False
            }

        # 运动员
        try:
            athlete_votes = df[df['is_athlete']]['fan_vote']
            non_athlete_votes = df[~df['is_athlete']]['fan_vote']
            if len(athlete_votes) > 0 and len(non_athlete_votes) > 0:
                athlete_test = stats.ttest_ind(athlete_votes, non_athlete_votes)
                results['athlete'] = {
                    'mean_diff': athlete_votes.mean() - non_athlete_votes.mean(),
                    'p_value': athlete_test.pvalue,
                    'significant': athlete_test.pvalue < 0.05
                }
            else:
                results['athlete'] = {
                    'mean_diff': 0.0,
                    'p_value': 1.0,
                    'significant': False
                }
        except:
            results['athlete'] = {
                'mean_diff': 0.0,
                'p_value': 1.0,
                'significant': False
            }

        # 歌手
        try:
            singer_votes = df[df['is_singer']]['fan_vote']
            non_singer_votes = df[~df['is_singer']]['fan_vote']
            if len(singer_votes) > 0 and len(non_singer_votes) > 0:
                singer_test = stats.ttest_ind(singer_votes, non_singer_votes)
                results['singer'] = {
                    'mean_diff': singer_votes.mean() - non_singer_votes.mean(),
                    'p_value': singer_test.pvalue,
                    'significant': singer_test.pvalue < 0.05
                }
            else:
                results['singer'] = {
                    'mean_diff': 0.0,
                    'p_value': 1.0,
                    'significant': False
                }
        except:
            results['singer'] = {
                'mean_diff': 0.0,
                'p_value': 1.0,
                'significant': False
            }

        # 演员
        try:
            actor_votes = df[df['is_actor']]['fan_vote']
            non_actor_votes = df[~df['is_actor']]['fan_vote']
            if len(actor_votes) > 0 and len(non_actor_votes) > 0:
                actor_test = stats.ttest_ind(actor_votes, non_actor_votes)
                results['actor'] = {
                    'mean_diff': actor_votes.mean() - non_actor_votes.mean(),
                    'p_value': actor_test.pvalue,
                    'significant': actor_test.pvalue < 0.05
                }
            else:
                results['actor'] = {
                    'mean_diff': 0.0,
                    'p_value': 1.0,
                    'significant': False
                }
        except:
            results['actor'] = {
                'mean_diff': 0.0,
                'p_value': 1.0,
                'significant': False
            }

        return results

    def _compare_impacts(self, judge_impact: Dict, fan_impact: Dict) -> Dict:
        """对比特征对judge和fan的不同影响（v2.0 - 修正比率计算）"""
        comparison = {}

        for feature in judge_impact.keys():
            judge_effect = judge_impact[feature]
            fan_effect = fan_impact[feature]

            if 'correlation' in judge_effect:
                # 相关系数比较
                comparison[feature] = {
                    'judge_correlation': judge_effect['correlation'],
                    'fan_correlation': fan_effect['correlation'],
                    'difference': abs(judge_effect['correlation'] - fan_effect['correlation']),
                    'judge_significant': judge_effect['significant'],
                    'fan_significant': fan_effect['significant']
                }
            else:
                # 均值差异比较（修正版）
                judge_diff = judge_effect['mean_diff']
                fan_diff = fan_effect['mean_diff']

                # 标准化效应量（Cohen's d）
                judge_std = 3.0  # 评委分标准差约为3分
                fan_std = 500000  # 粉丝票标准差约为50万

                judge_effect_size = judge_diff / judge_std
                fan_effect_size = fan_diff / fan_std

                # 计算比率（使用标准化效应量）
                if abs(judge_effect_size) > 0.01:
                    ratio = fan_effect_size / judge_effect_size
                else:
                    ratio = 0.0

                # 限制比率范围
                ratio = np.clip(ratio, -10, 10)

                comparison[feature] = {
                    'judge_effect': judge_diff,
                    'fan_effect': fan_diff,
                    'judge_effect_size': judge_effect_size,
                    'fan_effect_size': fan_effect_size,
                    'ratio': ratio,
                    'judge_significant': judge_effect['significant'],
                    'fan_significant': fan_effect['significant']
                }

        return comparison

    def generate_impact_report(self, analysis: Dict) -> pd.DataFrame:
        """生成特征影响报告表（v2.0 - 使用效应量）"""
        comparison = analysis['comparison']

        rows = []
        for feature, data in comparison.items():
            if 'correlation' in data:
                # 对于相关系数类型的特征（如age）
                rows.append({
                    'Feature': feature.capitalize(),
                    'Judge Effect': f"r={data['judge_correlation']:.3f}",
                    'Judge Sig': '✓' if data['judge_significant'] else '✗',
                    'Fan Effect': f"r={data['fan_correlation']:.3f}",
                    'Fan Sig': '✓' if data['fan_significant'] else '✗',
                    'Metric': f"Δ={data['difference']:.3f}"
                })
            else:
                # 对于均值差异类型的特征（使用效应量）
                judge_es = data.get('judge_effect_size', 0)
                fan_es = data.get('fan_effect_size', 0)
                ratio = data.get('ratio', 0)

                rows.append({
                    'Feature': feature.capitalize(),
                    'Judge Effect': f"d={judge_es:.2f}",  # Cohen's d
                    'Judge Sig': '✓' if data.get('judge_significant', False) else '✗',
                    'Fan Effect': f"d={fan_es:.2f}",  # Cohen's d
                    'Fan Sig': '✓' if data.get('fan_significant', False) else '✗',
                    'Metric': f"{ratio:.2f}x" if abs(ratio) < 10 else 'N/A'
                })

        return pd.DataFrame(rows)

    def plot_feature_comparison(self,
                               analysis: Dict,
                               save_path: str = None):
        """绘制特征影响对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        comparison = analysis['comparison']
        df = analysis['data']

        # 子图1: 年龄影响对比
        ax1 = axes[0, 0]
        ax1.scatter(df['age'], df['judge_score'], alpha=0.5, s=50,
                   label='Judge Score', color='#D62246')
        ax1_twin = ax1.twinx()
        ax1_twin.scatter(df['age'], df['fan_vote']/1e6, alpha=0.5, s=50,
                        label='Fan Votes (M)', color='#06A77D', marker='s')

        # 添加趋势线
        try:
            z_judge = np.polyfit(df['age'], df['judge_score'], 1)
            p_judge = np.poly1d(z_judge)
            ax1.plot(df['age'], p_judge(df['age']), "--", color='#D62246', linewidth=2)

            z_fan = np.polyfit(df['age'], df['fan_vote']/1e6, 1)
            p_fan = np.poly1d(z_fan)
            ax1_twin.plot(df['age'], p_fan(df['age']), "--", color='#06A77D', linewidth=2)
        except:
            pass

        ax1.set_xlabel('Age', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Judge Score', fontsize=12, fontweight='bold', color='#D62246')
        ax1_twin.set_ylabel('Fan Votes (M)', fontsize=12, fontweight='bold', color='#06A77D')
        ax1.set_title(f'Age Impact\nJudge r={comparison["age"]["judge_correlation"]:.3f}, '
                     f'Fan r={comparison["age"]["fan_correlation"]:.3f}',
                     fontsize=13, fontweight='bold')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 子图4: 影响比率对比（使用效应量）
        ax4 = axes[1, 1]
        categories = ['Athlete', 'Singer', 'Actor']

        # 使用效应量而非原始比率
        judge_effects = [comparison[cat.lower()]['judge_effect_size'] for cat in categories]
        fan_effects = [comparison[cat.lower()]['fan_effect_size'] for cat in categories]

        x = np.arange(len(categories))
        width = 0.35

        bars1 = ax4.bar(x - width / 2, judge_effects, width,
                        label='Judge Effect', color='#D62246', alpha=0.7, edgecolor='black')
        bars2 = ax4.bar(x + width / 2, fan_effects, width,
                        label='Fan Effect', color='#06A77D', alpha=0.7, edgecolor='black')

        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.axhline(y=0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Small Effect')
        ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Medium Effect')
        ax4.axhline(y=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Large Effect')

        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.set_ylabel("Effect Size (Cohen's d)", fontsize=12, fontweight='bold')
        ax4.set_title("Celebrity Type Impact: Standardized Effect Sizes\n(Judges vs Fans)",
                      fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if abs(height) > 0.05:
                    ax4.text(bar.get_x() + bar.get_width() / 2., height,
                             f'{height:.2f}', ha='center',
                             va='bottom' if height > 0 else 'top',
                             fontsize=9, fontweight='bold')

        # 子图3: 职业类型影响（Fan）
        ax3 = axes[1, 0]
        fan_effects = [comparison[cat.lower()]['fan_effect']/1e6 for cat in categories]
        colors_fan = ['#06A77D' if e > 0 else '#D62246' for e in fan_effects]

        bars = ax3.barh(categories, fan_effects, color=colors_fan, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax3.set_xlabel('Effect on Fan Votes (Millions)', fontsize=12, fontweight='bold')
        ax3.set_title('Celebrity Type Impact on Fans', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')

        # 添加显著性标记
        for i, (cat, effect) in enumerate(zip(categories, fan_effects)):
            if comparison[cat.lower()]['fan_significant']:
                ax3.text(effect, i, ' *', fontsize=20, va='center',
                        ha='left' if effect > 0 else 'right')

        # 子图4: 影响比率对比
        ax4 = axes[1, 1]
        ratios = [comparison[cat.lower()]['ratio'] for cat in categories]
        ratios = [min(max(r, -10), 10) for r in ratios]  # 限制范围

        x = np.arange(len(categories))
        bars = ax4.bar(x, ratios, color='#2E86AB', alpha=0.7, edgecolor='black')
        ax4.axhline(y=1, color='red', linestyle='--', linewidth=2,
                   label='Equal Impact', alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.set_ylabel('Fan Effect / Judge Effect', fontsize=12, fontweight='bold')
        ax4.set_title('Relative Impact: Fans vs Judges\n(>1 = Fans more influenced)',
                     fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # 添加数值标签
        for i, (bar, ratio) in enumerate(zip(bars, ratios)):
            height = bar.get_height()
            if abs(height) > 0.1:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{ratio:.2f}x', ha='center',
                        va='bottom' if height > 0 else 'top',
                        fontsize=10, fontweight='bold')

        plt.suptitle('Feature Impact Analysis: Judges vs Fans\n(* = statistically significant, p<0.05)',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()