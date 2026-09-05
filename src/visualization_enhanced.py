"""
增强版可视化模块 - 高级配色与美观布局
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from scipy import stats

# 设置高级样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("notebook", font_scale=1.2)

# 自定义配色方案
COLORS = {
    'primary': '#2E86AB',  # 深蓝
    'secondary': '#A23B72',  # 紫红
    'accent': '#F18F01',  # 橙色
    'success': '#06A77D',  # 绿色
    'warning': '#D62246',  # 红色
    'neutral': '#6C757D',  # 灰色
    'light': '#E8F4F8',  # 浅蓝
    'dark': '#2C3E50'  # 深灰
}

PALETTE = [COLORS['primary'], COLORS['secondary'], COLORS['accent'],
           COLORS['success'], COLORS['warning'], COLORS['neutral']]


class EnhancedDWTSVisualizer:
    """增强版DWTS可视化器"""

    def __init__(self, figsize=(14, 8)):
        self.figsize = figsize
        sns.set_palette(PALETTE)

    def plot_fan_vote_estimates_enhanced(self,
                                         result: Dict,
                                         save_path: str = None):
        """增强版粉丝票数估计图"""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        contestants = result['contestants']
        mean = result['fan_vote_mean']
        lower = result['fan_vote_lower']
        upper = result['fan_vote_upper']
        judge_scores = result['judge_scores']

        # 子图1: 票数估计（主图）
        ax1 = fig.add_subplot(gs[0, :])
        x = np.arange(len(contestants))

        # 渐变色条形图
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(contestants)))
        bars = ax1.bar(x, mean, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

        # 误差线
        ax1.errorbar(x, mean,
                     yerr=[mean - lower, upper - mean],
                     fmt='none',
                     ecolor='black',
                     capsize=8,
                     capthick=2,
                     alpha=0.6)

        # 标记被淘汰者
        eliminated_idx = contestants.index(result['eliminated'])
        ax1.scatter(eliminated_idx, mean[eliminated_idx],
                    color=COLORS['warning'], s=500, marker='X',
                    label='Eliminated', zorder=10, edgecolors='black', linewidth=2)

        # 添加数值标签
        for i, (m, l, u) in enumerate(zip(mean, lower, upper)):
            ax1.text(i, m + (u - m) * 1.1, f'{m / 1e6:.2f}M',
                     ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax1.set_xticks(x)
        ax1.set_xticklabels(contestants, rotation=45, ha='right', fontsize=11)
        ax1.set_ylabel('Estimated Fan Votes', fontsize=13, fontweight='bold')
        ax1.set_title(f'Fan Vote Estimates - Season {result["season"]} Week {result["week"]}',
                      fontsize=16, fontweight='bold', pad=20)
        ax1.legend(fontsize=12, loc='upper right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_facecolor(COLORS['light'])

        # 子图2: 评委分数对比
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.barh(contestants, judge_scores, color=COLORS['primary'], alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Judge Score', fontsize=12, fontweight='bold')
        ax2.set_title('Judge Scores', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_facecolor(COLORS['light'])

        # 子图3: 不确定性分析
        ax3 = fig.add_subplot(gs[1, 1])
        uncertainty = (upper - lower) / mean * 100
        bars3 = ax3.bar(range(len(contestants)), uncertainty,
                        color=COLORS['accent'], alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(contestants)))
        ax3.set_xticklabels(range(1, len(contestants) + 1))
        ax3.set_xlabel('Contestant #', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Uncertainty (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Estimation Uncertainty', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_facecolor(COLORS['light'])

        # 添加平均线
        ax3.axhline(uncertainty.mean(), color=COLORS['warning'],
                    linestyle='--', linewidth=2, label=f'Mean: {uncertainty.mean():.1f}%')
        ax3.legend()

        plt.suptitle('Comprehensive Fan Vote Analysis',
                     fontsize=18, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_system_comparison_enhanced(self,
                                        comparison_summary: List[Dict],
                                        save_path: str = None):
        """增强版系统比较图"""
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

        systems = ['Rank', 'Percentage', 'Rank + Judge Save', 'Percentage + Judge Save']

        # 收集数据
        data = {system: [] for system in systems}
        weeks = []

        for comp in comparison_summary:
            weeks.append(comp['season_week'])
            for system in systems:
                data[system].append(comp['comparison'][system]['correct_prob'])

        # 子图1-4: 各系统的时间序列
        for idx, system in enumerate(systems):
            row = idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])

            probs = data[system]
            x = range(len(probs))

            # 面积图
            ax.fill_between(x, probs, alpha=0.3, color=PALETTE[idx])
            ax.plot(x, probs, marker='o', linewidth=2.5,
                    markersize=8, color=PALETTE[idx], label=system)

            # 添加趋势线
            z = np.polyfit(x, probs, 2)
            p = np.poly1d(z)
            ax.plot(x, p(x), "--", alpha=0.5, color=COLORS['dark'], linewidth=2)

            # 参考线
            ax.axhline(0.5, color=COLORS['warning'], linestyle=':',
                       linewidth=2, alpha=0.5, label='Random Guess')
            ax.axhline(np.mean(probs), color=COLORS['success'],
                       linestyle='--', linewidth=2, alpha=0.7,
                       label=f'Mean: {np.mean(probs):.1%}')

            ax.set_xticks(x[::2])
            ax.set_xticklabels([weeks[i] for i in x[::2]], rotation=45, ha='right')
            ax.set_ylabel('Correct Prediction Probability', fontsize=11, fontweight='bold')
            ax.set_title(system, fontsize=13, fontweight='bold', pad=10)
            ax.set_ylim(-0.05, 1.1)
            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_facecolor(COLORS['light'])

        # 子图5: 系统总体对比（箱线图）
        ax5 = fig.add_subplot(gs[2, :])

        box_data = [data[system] for system in systems]
        bp = ax5.boxplot(box_data, labels=systems, patch_artist=True,
                         notch=True, showmeans=True,
                         boxprops=dict(facecolor=COLORS['light'], alpha=0.7),
                         medianprops=dict(color=COLORS['warning'], linewidth=2),
                         meanprops=dict(marker='D', markerfacecolor=COLORS['success'],
                                        markersize=8))

        # 为每个箱子设置不同颜色
        for patch, color in zip(bp['boxes'], PALETTE):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # 添加散点
        for i, system_data in enumerate(box_data):
            y = system_data
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax5.scatter(x, y, alpha=0.4, s=30, color=PALETTE[i])

        ax5.set_ylabel('Correct Prediction Probability', fontsize=13, fontweight='bold')
        ax5.set_title('Overall System Performance Comparison',
                      fontsize=14, fontweight='bold', pad=15)
        ax5.grid(True, alpha=0.3, axis='y')
        ax5.set_facecolor(COLORS['light'])
        ax5.set_xticklabels(systems, rotation=15, ha='right')

        plt.suptitle('Voting System Performance Analysis',
                     fontsize=18, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_feature_importance_enhanced(self,
                                         trace,
                                         feature_names: List[str],
                                         save_path: str = None):
        """增强版特征重要性图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 提取beta样本
        if isinstance(trace, dict):
            beta_samples = trace['posterior']['beta']
        else:
            beta_samples = trace.posterior['beta'].values
            beta_samples = beta_samples.reshape(-1, beta_samples.shape[-1])

        # 计算统计量
        beta_mean = beta_samples.mean(axis=0)
        beta_std = beta_samples.std(axis=0)
        beta_lower = np.percentile(beta_samples, 2.5, axis=0)
        beta_upper = np.percentile(beta_samples, 97.5, axis=0)

        # 子图1: 森林图（增强版）
        y_pos = np.arange(len(feature_names))

        # 根据均值排序
        sorted_idx = np.argsort(np.abs(beta_mean))[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_mean = beta_mean[sorted_idx]
        sorted_lower = beta_lower[sorted_idx]
        sorted_upper = beta_upper[sorted_idx]

        # 颜色编码（正负不同颜色）
        colors = [COLORS['success'] if m > 0 else COLORS['warning'] for m in sorted_mean]

        ax1.errorbar(sorted_mean, y_pos,
                     xerr=[sorted_mean - sorted_lower, sorted_upper - sorted_mean],
                     fmt='o', markersize=10, capsize=8, capthick=2,
                     ecolor='gray', alpha=0.7)

        # 为每个点单独上色
        for i, (m, c) in enumerate(zip(sorted_mean, colors)):
            ax1.scatter(m, i, s=200, color=c, zorder=5, edgecolors='black', linewidth=1.5)

        # 零线
        ax1.axvline(x=0, color=COLORS['dark'], linestyle='--', linewidth=2, alpha=0.5)

        # 显著性区域
        ax1.axvspan(-0.5, 0.5, alpha=0.1, color='gray', label='Non-significant zone')

        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(sorted_names, fontsize=12)
        ax1.set_xlabel('Coefficient Value (β)', fontsize=13, fontweight='bold')
        ax1.set_title('Feature Importance (Forest Plot)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.set_facecolor(COLORS['light'])
        ax1.legend()

        # 子图2: 后验分布（小提琴图）
        parts = ax2.violinplot([beta_samples[:, i] for i in sorted_idx],
                               positions=y_pos,
                               vert=False,
                               widths=0.7,
                               showmeans=True,
                               showmedians=True)

        # 自定义小提琴图颜色
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
            pc.set_edgecolor('black')

        ax2.axvline(x=0, color=COLORS['dark'], linestyle='--', linewidth=2, alpha=0.5)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(sorted_names, fontsize=12)
        ax2.set_xlabel('Coefficient Value (β)', fontsize=13, fontweight='bold')
        ax2.set_title('Posterior Distributions', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.set_facecolor(COLORS['light'])

        plt.suptitle('Feature Importance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_dwr_analysis_enhanced(self,
                                   historical_stds: np.ndarray,
                                   dwr_system,
                                   all_results: List[Dict],
                                   save_path: str = None):
        """增强版DWR分析图"""
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 子图1: 标准差分布（增强版）
        ax1 = fig.add_subplot(gs[0, 0])
        n, bins, patches = ax1.hist(historical_stds, bins=25, alpha=0.7,
                                    edgecolor='black', linewidth=1.5)

        # 渐变色
        cm = plt.cm.RdYlGn_r
        for i, patch in enumerate(patches):
            patch.set_facecolor(cm(i / len(patches)))

        ax1.axvline(dwr_system.historical_std, color=COLORS['warning'],
                    linestyle='--', linewidth=3,
                    label=f'Mean: {dwr_system.historical_std:.2f}')
        ax1.set_xlabel('Judge Score Std Dev', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Historical Variability', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor(COLORS['light'])

        # 子图2: 权重函数（3D效果）
        ax2 = fig.add_subplot(gs[0, 1])
        std_range = np.linspace(0, max(historical_stds) * 1.3, 200)
        weights = []

        for std in std_range:
            w = 0.5 + dwr_system.k * (dwr_system.historical_std - std)
            w = np.clip(w, 0.2, 0.8)
            weights.append(w)

        # 渐变填充
        ax2.fill_between(std_range, 0.2, weights, alpha=0.3, color=COLORS['primary'])
        ax2.fill_between(std_range, weights, 0.8, alpha=0.3, color=COLORS['accent'])
        ax2.plot(std_range, weights, linewidth=3, color=COLORS['dark'])

        # 关键点标注
        ax2.scatter([dwr_system.historical_std], [0.5],
                    s=200, color=COLORS['warning'], zorder=5,
                    edgecolors='black', linewidth=2, label='Equilibrium')

        ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(dwr_system.historical_std, color=COLORS['warning'],
                    linestyle='--', alpha=0.5)

        ax2.set_xlabel('Current Week Std Dev', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Judge Weight (w)', fontsize=12, fontweight='bold')
        ax2.set_title('Dynamic Weight Function', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor(COLORS['light'])

        # 子图3: 权重分布实例
        ax3 = fig.add_subplot(gs[0, 2])
        actual_weights = []
        for result in all_results[:15]:
            std = np.std(result['judge_scores'])
            w = 0.5 + dwr_system.k * (dwr_system.historical_std - std)
            w = np.clip(w, 0.2, 0.8)
            actual_weights.append(w)

        ax3.hist(actual_weights, bins=15, alpha=0.7, color=COLORS['success'],
                 edgecolor='black', linewidth=1.5)
        ax3.axvline(np.mean(actual_weights), color=COLORS['warning'],
                    linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(actual_weights):.2f}')
        ax3.set_xlabel('Judge Weight', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax3.set_title('Actual Weight Distribution', fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor(COLORS['light'])

        # 子图4-6: MAD分数分析
        ax4 = fig.add_subplot(gs[1, :])

        # 计算不同系统的MAD
        weeks = []
        mad_rank = []
        mad_pct = []
        mad_dwr = []

        for result in all_results[:15]:
            weeks.append(f"S{result['season']}W{result['week']}")

            fan_mean = result['fan_vote_mean']
            judge_scores = result['judge_scores']

            N = len(fan_mean)
            judge_ranks = N - np.argsort(np.argsort(judge_scores))
            fan_ranks = N - np.argsort(np.argsort(fan_mean))

            # Rank系统
            combined_ranks = judge_ranks + fan_ranks
            final_ranks = N - np.argsort(np.argsort(combined_ranks))
            mad_rank.append(np.sum((final_ranks - judge_ranks) ** 2 +
                                   (final_ranks - fan_ranks) ** 2))

            # Percentage系统
            j_pct = judge_scores / judge_scores.sum()
            f_pct = fan_mean / fan_mean.sum()
            combined_pct = j_pct + f_pct
            final_ranks_pct = N - np.argsort(np.argsort(combined_pct))
            mad_pct.append(np.sum((final_ranks_pct - judge_ranks) ** 2 +
                                  (final_ranks_pct - fan_ranks) ** 2))

            # DWR系统
            final_scores, w = dwr_system.rank_contestants(judge_scores, fan_mean)
            final_ranks_dwr = N - np.argsort(np.argsort(final_scores))
            mad = dwr_system.calculate_MAD(judge_ranks, fan_ranks, final_ranks_dwr, w)
            mad_dwr.append(mad)

        x = range(len(weeks))
        width = 0.25

        ax4.bar([i - width for i in x], mad_rank, width,
                label='Rank', color=PALETTE[0], alpha=0.8, edgecolor='black')
        ax4.bar(x, mad_pct, width,
                label='Percentage', color=PALETTE[1], alpha=0.8, edgecolor='black')
        ax4.bar([i + width for i in x], mad_dwr, width,
                label='DWR (Proposed)', color=PALETTE[3], alpha=0.8, edgecolor='black')

        ax4.set_xticks(x)
        ax4.set_xticklabels(weeks, rotation=45, ha='right')
        ax4.set_ylabel('MAD Score (Lower is Better)', fontsize=12, fontweight='bold')
        ax4.set_title('System Fairness Comparison (MAD Metric)',
                      fontsize=14, fontweight='bold')
        ax4.legend(fontsize=11, loc='upper right')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_facecolor(COLORS['light'])

        # 添加平均值标注
        for i, (r, p, d) in enumerate(zip(mad_rank, mad_pct, mad_dwr)):
            if i % 3 == 0:  # 每3个标注一次，避免拥挤
                ax4.text(i - width, r, f'{r:.1f}', ha='center', va='bottom', fontsize=8)
                ax4.text(i, p, f'{p:.1f}', ha='center', va='bottom', fontsize=8)
                ax4.text(i + width, d, f'{d:.1f}', ha='center', va='bottom', fontsize=8)

        plt.suptitle('Dynamic Weighted Ranking (DWR) System Analysis',
                     fontsize=18, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_convergence_diagnostics(self,
                                     all_results: List[Dict],
                                     save_path: str = None):
        """收敛性诊断图"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 提取数据
        weeks = [f"S{r['season']}W{r['week']}" for r in all_results]
        rhats = [r['rhat'] for r in all_results]
        n_contestants = [len(r['contestants']) for r in all_results]
        mean_votes = [r['fan_vote_mean'].mean() for r in all_results]
        uncertainties = [((r['fan_vote_upper'] - r['fan_vote_lower']) / r['fan_vote_mean']).mean()
                         for r in all_results]

        # 子图1: R-hat值
        ax1 = axes[0, 0]
        ax1.scatter(range(len(rhats)), rhats, s=100, alpha=0.6,
                    c=PALETTE[0], edgecolors='black', linewidth=1.5)
        ax1.axhline(1.01, color=COLORS['warning'], linestyle='--',
                    linewidth=2, label='Convergence Threshold')
        ax1.set_ylabel('R-hat Value', fontsize=12, fontweight='bold')
        ax1.set_title('MCMC Convergence (R-hat)', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor(COLORS['light'])

        # 子图2: 选手数量 vs 不确定性
        ax2 = axes[0, 1]
        scatter = ax2.scatter(n_contestants, uncertainties, s=150,
                              c=range(len(n_contestants)), cmap='viridis',
                              alpha=0.6, edgecolors='black', linewidth=1.5)
        ax2.set_xlabel('Number of Contestants', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Uncertainty (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Uncertainty vs Complexity', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=ax2, label='Week Index')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor(COLORS['light'])

        # 子图3: 平均票数趋势
        ax3 = axes[1, 0]
        ax3.plot(range(len(mean_votes)), mean_votes, marker='o',
                 linewidth=2.5, markersize=8, color=PALETTE[2])
        ax3.fill_between(range(len(mean_votes)), mean_votes, alpha=0.3, color=PALETTE[2])
        ax3.set_xlabel('Week Index', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Mean Fan Votes', fontsize=12, fontweight='bold')
        ax3.set_title('Fan Vote Trend', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.set_facecolor(COLORS['light'])

        # 子图4: 不确定性分布
        ax4 = axes[1, 1]
        ax4.hist(uncertainties, bins=15, alpha=0.7, color=PALETTE[4],
                 edgecolor='black', linewidth=1.5)
        ax4.axvline(np.mean(uncertainties), color=COLORS['warning'],
                    linestyle='--', linewidth=2,
                    label=f'Mean: {np.mean(uncertainties):.1f}%')
        ax4.set_xlabel('Uncertainty (%)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.set_title('Uncertainty Distribution', fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_facecolor(COLORS['light'])

        plt.suptitle('Model Convergence and Quality Diagnostics',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def plot_season_comparison(self,
                               all_results: List[Dict],
                               save_path: str = None):
        """赛季对比分析"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 按赛季分组
        seasons = {}
        for r in all_results:
            s = r['season']
            if s not in seasons:
                seasons[s] = []
            seasons[s].append(r)

        season_ids = sorted(seasons.keys())

        # 子图1: 各赛季平均票数
        ax1 = axes[0, 0]
        avg_votes = [np.mean([r['fan_vote_mean'].mean() for r in seasons[s]])
                     for s in season_ids]
        bars = ax1.bar(range(len(season_ids)), avg_votes,
                       color=PALETTE[:len(season_ids)], alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(season_ids)))
        ax1.set_xticklabels([f'S{s}' for s in season_ids])
        ax1.set_ylabel('Average Fan Votes', fontsize=12, fontweight='bold')
        ax1.set_title('Fan Engagement by Season', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_facecolor(COLORS['light'])

        # 子图2:
        # 子图2: 各赛季评委分数变异性
        ax2 = axes[0, 1]
        judge_stds = [np.mean([np.std(r['judge_scores']) for r in seasons[s]])
                      for s in season_ids]
        ax2.plot(range(len(season_ids)), judge_stds, marker='o',
                 linewidth=3, markersize=10, color=COLORS['secondary'])
        ax2.fill_between(range(len(season_ids)), judge_stds, alpha=0.3,
                         color=COLORS['secondary'])
        ax2.set_xticks(range(len(season_ids)))
        ax2.set_xticklabels([f'S{s}' for s in season_ids])
        ax2.set_ylabel('Judge Score Std Dev', fontsize=12, fontweight='bold')
        ax2.set_title('Judge Consensus by Season', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor(COLORS['light'])

        # 子图3: 各赛季选手数量分布
        ax3 = axes[1, 0]
        for i, s in enumerate(season_ids):
            n_contestants = [len(r['contestants']) for r in seasons[s]]
            ax3.scatter([i] * len(n_contestants), n_contestants,
                        s=100, alpha=0.6, color=PALETTE[i % len(PALETTE)],
                        edgecolors='black', linewidth=1)
        ax3.set_xticks(range(len(season_ids)))
        ax3.set_xticklabels([f'S{s}' for s in season_ids])
        ax3.set_ylabel('Number of Contestants', fontsize=12, fontweight='bold')
        ax3.set_title('Competition Size by Season', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_facecolor(COLORS['light'])

        # 子图4: 各赛季不确定性
        ax4 = axes[1, 1]
        uncertainties_by_season = []
        for s in season_ids:
            unc = [((r['fan_vote_upper'] - r['fan_vote_lower']) / r['fan_vote_mean']).mean()
                   for r in seasons[s]]
            uncertainties_by_season.append(unc)

        bp = ax4.boxplot(uncertainties_by_season, labels=[f'S{s}' for s in season_ids],
                         patch_artist=True, notch=True)
        for patch, color in zip(bp['boxes'], PALETTE[:len(season_ids)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax4.set_ylabel('Uncertainty (%)', fontsize=12, fontweight='bold')
        ax4.set_title('Estimation Uncertainty by Season', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.set_facecolor(COLORS['light'])

        plt.suptitle('Cross-Season Comparison Analysis',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()