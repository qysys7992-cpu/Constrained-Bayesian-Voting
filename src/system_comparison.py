import numpy as np
from typing import Dict, List
from scipy import stats


class VotingSystemComparator:
    """投票系统比较器"""

    def __init__(self, fan_vote_samples):
        """
        Args:
            fan_vote_samples: 可以是numpy数组或trace对象
        """
        # 兼容不同输入格式
        if isinstance(fan_vote_samples, dict):
            # 来自简化版trace
            self.samples = fan_vote_samples['posterior']['fan_votes']
        elif hasattr(fan_vote_samples, 'posterior'):
            # 来自PyMC trace
            samples = fan_vote_samples.posterior['fan_votes'].values
            self.samples = samples.reshape(-1, samples.shape[-1])
        else:
            # 直接是numpy数组
            self.samples = fan_vote_samples

    # ... 其余代码保持不变 ...

    def simulate_elimination(self,
                             judge_scores: np.ndarray,
                             method: str,
                             judge_save: bool = False) -> np.ndarray:
        """
        蒙特卡洛模拟淘汰结果

        Args:
            judge_scores: 评委分数
            method: 'rank' 或 'percentage'
            judge_save: 是否启用评委拯救

        Returns:
            elimination_probs: (n_contestants,) 每个选手被淘汰的概率
        """
        n_samples, n_contestants = self.samples.shape
        elimination_counts = np.zeros(n_contestants)

        for i in range(n_samples):
            fan_votes = self.samples[i]

            if method == 'rank':
                judge_ranks = self._scores_to_ranks(judge_scores)
                fan_ranks = self._votes_to_ranks(fan_votes)
                combined = judge_ranks + fan_ranks

                if judge_save:
                    # 找出得分最低的两人
                    bottom_two = np.argsort(combined)[-2:]
                    # 评委救下评委分更高的那个
                    eliminated = bottom_two[np.argmin(judge_scores[bottom_two])]
                else:
                    eliminated = np.argmax(combined)

            else:  # percentage
                judge_pcts = judge_scores / judge_scores.sum()
                fan_pcts = fan_votes / fan_votes.sum()
                combined = judge_pcts + fan_pcts

                if judge_save:
                    bottom_two = np.argsort(combined)[:2]
                    eliminated = bottom_two[np.argmin(judge_scores[bottom_two])]
                else:
                    eliminated = np.argmin(combined)

            elimination_counts[eliminated] += 1

        return elimination_counts / n_samples

    @staticmethod
    def _scores_to_ranks(scores):
        return len(scores) - np.argsort(np.argsort(scores))

    @staticmethod
    def _votes_to_ranks(votes):
        return len(votes) - np.argsort(np.argsort(votes))

    def compare_systems(self,
                        judge_scores: np.ndarray,
                        true_eliminated_idx: int) -> Dict:
        """
        比较不同系统的表现

        Returns:
            comparison_results: 包含各系统的淘汰概率分布
        """
        results = {}

        methods = [
            ('Rank', 'rank', False),
            ('Percentage', 'percentage', False),
            ('Rank + Judge Save', 'rank', True),
            ('Percentage + Judge Save', 'percentage', True)
        ]

        for name, method, judge_save in methods:
            probs = self.simulate_elimination(judge_scores, method, judge_save)
            results[name] = {
                'elimination_probs': probs,
                'correct_prob': probs[true_eliminated_idx],
                'predicted_eliminated': np.argmax(probs)
            }

        return results

    def calculate_fan_influence_index(self,
                                      judge_scores: np.ndarray) -> np.ndarray:
        """
        计算粉丝影响力指数
        = (仅粉丝排名 - 仅评委排名)的期望
        """
        judge_ranks = self._scores_to_ranks(judge_scores)

        fan_rank_samples = np.array([
            self._votes_to_ranks(self.samples[i])
            for i in range(len(self.samples))
        ])

        mean_fan_ranks = fan_rank_samples.mean(axis=0)
        influence_index = mean_fan_ranks - judge_ranks

        return influence_index


# 使用示例
if __name__ == "__main__":
    # 模拟后验样本
    fan_vote_samples = np.random.lognormal(mean=10, sigma=1, size=(1000, 4))
    judge_scores = np.array([25, 28, 30, 27])

    comparator = VotingSystemComparator(fan_vote_samples)
    results = comparator.compare_systems(judge_scores, true_eliminated_idx=0)

    print("系统比较结果:")
    for system, data in results.items():
        print(f"\n{system}:")
        print(f"  淘汰概率: {data['elimination_probs']}")
        print(f"  预测被淘汰: 选手{data['predicted_eliminated']}")
        print(f"  正确概率: {data['correct_prob']:.2%}")