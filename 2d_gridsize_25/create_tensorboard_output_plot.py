import numpy as np
from matplotlib import pyplot as plt
from robot_state_utils import RobotStateUtils
from numpy import savetxt
import pandas as pd

if __name__ == '__main__':
    file_dir = "/home/vignesh/PycharmProjects/motion_planning_max_entropy_irl/2d_gridsize_25/"
    csv_name = "run-.-tag-input_info_rewards.csv"
    # csv_name = "run-.-tag-episode_reward.csv"
    trajectory_data_values = pd.read_csv(file_dir + csv_name).to_numpy()
    print(trajectory_data_values[0], trajectory_data_values.shape)
    plt.rcParams.update({'font.size': 12})
    step_val = []
    reward_val = []
    for i in range(trajectory_data_values.shape[0]):
        step_val.append(trajectory_data_values[i, 1])
        reward_val.append(trajectory_data_values[i, 2])

    plt.plot(step_val, reward_val)
    plt.xlabel('Steps')
    plt.ylabel('Reward Value')
    # plt.axis([0, 100000, -1, -0.2])
    plt.xticks([0, 500000, 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000],
               ["0", "500k", "1M", "1.5M", "2M", "2.5M", "3M", "3.5M", "4M"])
    # plt.yticks([-1, -0.8, -0.6, -0.4, -0.2])
    # plt.axis([0, 20000, 40000, 60000, 80000, 100000], [-1, -0.2])
    plt.savefig(file_dir + "rewards_plot.png")
    plt.show()
    # plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
    # ax = fig.add_subplot(11, projection='2d')