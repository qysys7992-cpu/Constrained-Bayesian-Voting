import numpy as np
from typing import Tuple
from typing import Tuple, List

class DynamicWeightedRanking:
    """动态加权排名系统(DWR)"""

    def __init__(self, k: float = 0.3):
        """
        Args:
            k: 调节系数,控制权重变化的敏感度
        """
        self.k = k
        self.historical_std = None

    def fit_historical_std(self, all_judge_scores: List[np.ndarray]):
        """
        从历史数据中学习评委分数的平均标准差

        Args:
            all_judge_scores: 历史所有周的评委分数列表
        """
        stds = [np.std(scores) for scores in all_judge_scores]
        self.historical_std = np.mean(stds)

    def calculate_dynamic_weight(self, judge_scores: np.ndarray) -> float:
        """
        根据当晚评委分数的一致性动态计算评委权重

        Args:
            judge_scores: 当晚的评委分数

        Returns:
            w: 评委权重 (0 <= w <= 1)
        """
        if self.historical_std is None:
            raise ValueError("必须先调用fit_historical_std()")

        current_std = np.std(judge_scores)

        # 标准差越小,评委意见越一致,权重越高
        w = 0.5 + self.k * (self.historical_std - current_std)

        # 限制在[0.2, 0.8]范围内
        w = np.clip(w, 0.2, 0.8)

        return w

    def rank_contestants(self,
                         judge_scores: np.ndarray,
                         fan_votes: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        使用DWR系统对选手排名

        Returns:
            final_scores: 最终得分
            weight_used: 使用的评委权重
        """
        w = self.calculate_dynamic_weight(judge_scores)

        # 标准化到[0,1]
        judge_norm = (judge_scores - judge_scores.min()) / (judge_scores.max() - judge_scores.min() + 1e-8)
        fan_norm = (fan_votes - fan_votes.min()) / (fan_votes.max() - fan_votes.min() + 1e-8)

        final_scores = w * judge_norm + (1 - w) * fan_norm

        return final_scores, w

    def calculate_MAD(self,
                      judge_ranks: np.ndarray,
                      fan_ranks: np.ndarray,
                      final_ranks: np.ndarray,
                      w: float) -> float:
        """
        计算最小化综合分歧(MAD)指标

        Args:
            judge_ranks, fan_ranks, final_ranks: 各种排名
            w: 评委权重

        Returns:
            mad_score: MAD分数(越小越好)
        """
        judge_disagreement = np.sum((final_ranks - judge_ranks) ** 2)
        fan_disagreement = np.sum((final_ranks - fan_ranks) ** 2)

        mad = w * judge_disagreement + (1 - w) * fan_disagreement

        return mad


# 使用示例
if __name__ == "__main__":
    # 模拟历史数据
    historical_scores = [
        np.array([25, 28, 30, 27, 26]),
        np.array([24, 29, 28, 26]),
        np.array([27, 27, 28, 29, 25])
    ]

    dwr = DynamicWeightedRanking(k=0.3)
    dwr.fit_historical_std(historical_scores)

    # 当晚数据
    tonight_judges = np.array([25, 28, 30, 27])
    tonight_fans = np.array([1000000, 1500000, 800000, 1200000])

    final_scores, weight = dwr.rank_contestants(tonight_judges, tonight_fans)

    print(f"动态权重: {weight:.2f}")
    print(f"最终得分: {final_scores}")
    print(f"最终排名: {np.argsort(final_scores)[::-1] + 1}")