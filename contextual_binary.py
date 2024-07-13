import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import expit

from bandit.bernoulli_ts import BernoulliTS
from bandit.lin_ts import LinTS
from bandit.lin_ucb import LinUCB
from bandit.lin_ucb_hybrid import LinUCBHybrid
from bandit.logistic_ts import LogisticTS
from bandit.logistic_pgts import LogisticPGTS
from bandit.bandit_base.bandit import BanditBase

def swap_dict_values_cyclic1(original_dict):

    keys = list(original_dict.keys())
    values = list(original_dict.values())
    
    # サイクルシフトして値を入れ替える
    tmp = values[3]
    values[3] = values[2]
    values[2] = values[1]
    values[1] = values[0]
    values[0] = tmp
    swapped_dict = dict(zip(keys, values))
    return swapped_dict

def swap_dict_values_cyclic2(original_dict):
    
    keys = list(original_dict.keys())
    values = list(original_dict.values())
    
    # サイクルシフトして値を入れ替える
    tmp = values[2]
    tmp2 = values[3]
    values[3] = values[1]
    values[1] = tmp2
    values[2] = values[0]
    values[0] = tmp
    swapped_dict = dict(zip(keys, values))
    return swapped_dict

def swap_dict_values_cyclic3(original_dict):
    
    keys = list(original_dict.keys())
    values = list(original_dict.values())
    
    # サイクルシフトして値を入れ替える
    tmp = values[1]
    tmp2 = values[2]
    tmp3 = values[3]
    values[3] = values[0]
    values[2] = tmp3
    values[1] = tmp2
    values[0] = tmp
    swapped_dict = dict(zip(keys, values))
    return swapped_dict


def get_batch(
    bandit: BanditBase,
    true_theta: dict[str, np.ndarray],
    features: list[str],
    batch_size: int,
    # swap_prob: bool = False,
    swap_func = None,
) -> pd.DataFrame:
    # 学習データ
    log = []
    for _ in range(batch_size):
        # len(feature) 個の乱数を生成
        x = np.random.rand(len(features))
        # 腕の選択
        arm_id = bandit.select_arm(x)
        # 報酬期待値
        true_prob = {a: expit(theta @ x) for a, theta in true_theta.items()}
        maxprob = max(true_prob.values())

        # true_prob を swap
        # if _ == batch_size / 2:
        #     true_prob = swap_dict_values_cyclic(true_prob)
        #     reward = np.random.binomial(1, true_prob[arm_id])
        # else:
        #     # true_prob = swap_dict_values_cyclic(true_prob)
        #     reward = np.random.binomial(1, true_prob[arm_id])
        if swap_func:
            true_prob = swap_func(true_prob)
        
        maxprob = max(true_prob.values())

        log.append(
            {
                "arm_id": arm_id,
                # n=1, p=true_prob[arm_id]) の二項分布
                "reward": np.random.binomial(1, true_prob[arm_id]),
                # "reward": np.random.randn(),
                # "reward": reward,
                "regret": maxprob - true_prob[arm_id],
            }
            | dict(zip(features, x))
        )
    return pd.DataFrame(log)

if __name__ == "__main__":
    batch_size = 1
    arm_num = 4
    feature_num = 5
    intercept = False
    arm_ids = [f"arm{i}" for i in range(arm_num)]
    features = [f"feat{i}" for i in range(feature_num)]
    true_theta = {a: np.random.normal(size=feature_num) for a in arm_ids}
    print(true_theta)

    for run in range(10):  # 10回繰り返す
        report = {}
        for bandit in [
            LogisticTS(arm_ids, features, intercept),
            LogisticPGTS(arm_ids, features, intercept, M=10),
            LinTS(arm_ids, features, intercept),
            LinUCB(arm_ids, features, intercept, alpha=1),
            LinUCBHybrid(arm_ids, features, intercept, alpha=1),
            # BernoulliTS(arm_ids),
        ]:
            name = bandit.__class__.__name__
            print(name)
            regret_log = []
            cumsum_regret = 0

            # 学習回数
            for i in tqdm(range(10000)):
                # true_probをswapするフラグを設定
                swap_func = None
                if 2500 <= i <= 5000:
                    swap_func = swap_dict_values_cyclic1
                elif 5000 <= i <= 7500:
                    swap_func = swap_dict_values_cyclic2
                elif 7500 <= i :
                    swap_func = swap_dict_values_cyclic3

                reward_df = get_batch(bandit, true_theta, features, batch_size, swap_func=swap_func)
                cumsum_regret += reward_df["regret"].sum()
                regret_log.append(cumsum_regret)
                bandit.train(reward_df)
            report[name] = regret_log

        # 結果をCSVファイルに書き出す
        report_df = pd.DataFrame(report)
        report_df.to_csv(f'bandit_results_run_{run + 1}.csv', index=False)