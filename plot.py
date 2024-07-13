import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_average_and_plot(num_runs, filename_template):
    all_reports = []

    for run in range(num_runs):
        report_df = pd.read_csv(filename_template.format(run + 1))
        all_reports.append(report_df)

    # 平均を計算
    average_report = pd.concat(all_reports).groupby(level=0).mean()

    # 結果のプロット
    average_report.plot()
    plt.xlabel("Batch Iteration")
    plt.ylabel("Cumulative Regret")
    plt.title("Contextual Binary Reward Bandit: Average Regret over Multiple Runs")
    plt.show()


if __name__ == "__main__":
    num_runs = 10
    filename_template = 'bandit_results_run_{}.csv'
    calculate_average_and_plot(num_runs, filename_template)