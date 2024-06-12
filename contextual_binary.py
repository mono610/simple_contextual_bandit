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


def swap_dict_values_cyclic(original_dict):
    """
    original_dict: 値を入れ替える対象の辞書。
    
    入れ替えの順序: key0 → key1 → key2 → key0
    """
    keys = list(original_dict.keys())
    values = list(original_dict.values())
    
    # サイクルシフトして値を入れ替える
    tmp = values[2]
    values[2] = values[1]
    values[1] = values[0]
    values[0] = tmp
    swapped_dict = dict(zip(keys, values))
    return swapped_dict


def get_batch(
    bandit: BanditBase,
    true_theta: dict[str, np.ndarray],
    features: list[str],
    batch_size: int,
) -> pd.DataFrame:
    # 学習データ
    log = []
    for _ in range(batch_size):
        # len(feature) 個の乱数を生成
        # 変更必要あり
        x = np.random.rand(len(features))
        arm_id = bandit.select_arm(x)
        true_prob = {a: expit(theta @ x) for a, theta in true_theta.items()}
        maxprob = max(true_prob.values())

        # true_prob を swap
        if _ == batch_size / 2:
            true_prob = swap_dict_values_cyclic(true_prob)
            reward = np.random.binomial(1, true_prob[arm_id])
        else:
            # true_prob = swap_dict_values_cyclic(true_prob)
            reward = np.random.binomial(1, true_prob[arm_id])

        log.append(
            {
                "arm_id": arm_id,
                # n=1, p=true_prob[arm_id]) の二項分布
                # "reward": np.random.binomial(1, true_prob[arm_id]),
                # "reward": np.random.randn(),
                "reward": reward,
                "regret": maxprob - true_prob[arm_id],
            }
            | dict(zip(features, x))
        )
    return pd.DataFrame(log)


if __name__ == "__main__":
    batch_size = 100
    arm_num = 3
    feature_num = 5
    intercept = False
    arm_ids = [f"arm{i}" for i in range(arm_num)]
    features = [f"feat{i}" for i in range(feature_num)]
    true_theta = {a: np.random.normal(size=feature_num) for a in arm_ids}
    print(true_theta)

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
        # episode 数
        for i in tqdm(range(100)):
            # true_theta を swap
            # if i == 50:
            #     true_theta = swap_dict_values_cyclic(true_theta)

            reward_df = get_batch(bandit, true_theta, features, batch_size)
            cumsum_regret += reward_df["regret"].sum()
            regret_log.append(cumsum_regret)
            bandit.train(reward_df)
        report[name] = regret_log
    pd.DataFrame(report).plot()
    plt.xlabel("Batch Iteration")
    plt.ylabel("Cumulative Regret")
    plt.title(
        f"Contextual Binary Reward Bandit: batch_size={batch_size}, arm_num={arm_num}"
    )
    plt.show()