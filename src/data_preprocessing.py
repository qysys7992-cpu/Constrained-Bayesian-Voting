import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class DWTSDataProcessor:
    """DWTS数据预处理器"""

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.processed_data = []

    def get_week_columns(self, week: int) -> List[str]:
        """获取某周的评委分数列"""
        return [col for col in self.df.columns
                if col.startswith(f'week{week}_judge')]

    def calculate_total_judge_score(self, row: pd.Series, week: int) -> float:
        """计算某周的评委总分"""
        week_cols = self.get_week_columns(week)
        scores = [row[col] for col in week_cols if pd.notna(row[col])]
        return sum(scores) if scores else 0

    def extract_weekly_data(self) -> List[Dict]:
        """
        提取每周的比赛数据
        返回格式: [{
            'season': int,
            'week': int,
            'contestants': [contestant_info_dict],
            'eliminated': str
        }]
        """
        weekly_data = []

        for season in self.df['season'].unique():
            season_df = self.df[self.df['season'] == season]

            # 确定该赛季的周数
            max_week = 0
            for col in self.df.columns:
                if col.startswith('week') and '_judge' in col:
                    week_num = int(col.split('_')[0].replace('week', ''))
                    max_week = max(max_week, week_num)

            for week in range(1, max_week + 1):
                week_contestants = []
                eliminated_this_week = None

                for _, row in season_df.iterrows():
                    # 检查该选手在这周是否还在比赛
                    total_score = self.calculate_total_judge_score(row, week)

                    if total_score > 0:  # 还在比赛中
                        contestant_info = {
                            'name': row['celebrity_name'],
                            'partner': row['ballroom_partner'],
                            'industry': row['celebrity_industry'],
                            'age': row['celebrity_age_during_season'],
                            'judge_score': total_score,
                            'placement': row['placement']
                        }
                        week_contestants.append(contestant_info)

                    # 检查是否在本周被淘汰
                    if pd.notna(row['results']) and f'Week {week}' in str(row['results']):
                        eliminated_this_week = row['celebrity_name']

                # 只保存有淘汰的周
                if eliminated_this_week and len(week_contestants) >= 2:
                    weekly_data.append({
                        'season': int(season),
                        'week': week,
                        'contestants': week_contestants,
                        'eliminated': eliminated_this_week
                    })

        return weekly_data

    def create_feature_matrix(self, contestants: List[Dict]) -> np.ndarray:
        """
        为贝叶斯模型创建特征矩阵
        特征: [age, is_athlete, is_singer, avg_judge_score, partner_encoded]
        """
        features = []

        for c in contestants:
            age_normalized = (c['age'] - 30) / 15  # 标准化年龄
            is_athlete = 1 if 'Athlete' in str(c['industry']) else 0
            is_singer = 1 if 'Singer' in str(c['industry']) else 0
            judge_score_norm = c['judge_score'] / 30  # 假设满分30

            # 简化的舞伴编码(实际应该用更复杂的编码)
            partner_hash = hash(c['partner']) % 10 / 10

            features.append([
                age_normalized,
                is_athlete,
                is_singer,
                judge_score_norm,
                partner_hash
            ])

        return np.array(features)


# 使用示例
if __name__ == "__main__":
    processor = DWTSDataProcessor('2026_MCM_Problem_C_Data.csv')
    weekly_data = processor.extract_weekly_data()
    print(f"提取了 {len(weekly_data)} 周的数据")
    print(f"第一周示例: {weekly_data[0]}")