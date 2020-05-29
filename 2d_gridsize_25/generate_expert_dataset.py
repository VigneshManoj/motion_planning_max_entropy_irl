import pandas as pd
import numpy as np
import csv
import math
from numpy import linalg as LA
import matplotlib.pyplot as plt


def create_reward(data_values, goal_pos, file_dir):
    total_reward = 0
    list_of_reward = []
    epi_ret = []
    print(data_values.shape)
    for i in range(data_values.shape[0]):
        # dist_x = np.subtract(final_state, actual_val[i, 0])
        # dist_y = np.subtract(final_state, actual_val[i, 1])
        # dist_z = np.subtract(final_state, actual_val[i, 2])
        # reward = math.sqrt(math.pow(dist_x, 2) + math.pow(dist_y, 2) + math.pow(dist_z, 2))
        cur_dist = LA.norm(np.subtract(goal_pos, data_values[i, 0:3]))
        # reward = round(1 - float(abs(cur_dist)/0.05)*0.5, 5)
        reward = -cur_dist
        if abs(cur_dist) < 0.000001:
            reward = 1

        total_reward += reward
        list_of_reward.append(reward)
    epi_ret.append(total_reward)
    print(total_reward)
    # np.savetxt(file_dir + 'episode_returns.csv', total_reward, delimiter=',')
    # np.savetxt(file_dir + 'rewards.csv', np.array(list_of_reward), delimiter=',')
    # np.save(file_dir + 'rewards', np.array(list_of_reward))
    np.save(file_dir + 'episode_returns', epi_ret)


def create_obs_action_func(data_values, file_dir):
    obs = []
    extra_obs = np.zeros(17)
    actions = []
    extra_observation = []
    # print(data_values.shape)
    for i in range(data_values.shape[0]):
        extra_observation = data_values[i, 0:3].flatten().tolist() + extra_obs.tolist()
        # obs.append(data_values[i, 0:3])
        obs.append(extra_observation)
        actions.append(data_values[i, 3:6])
    print(np.array(obs).shape, np.array(obs))
    np.save(file_dir + 'obs', np.array(obs))
    # np.save(file_dir + 'actions', np.array(actions))


if __name__ == "__main__":
    # Assuming reward is reward = round(1 - float(abs(cur_dist)/0.3)*0.5, 5)
    file_dir = "/home/vignesh/Thesis_Suture_data/trial2/ambf_data/"
    csv_name = "832953.csv"
    trajectory_data_values = pd.read_csv(file_dir + csv_name).to_numpy()
    final_state = np.array([0.005, 0.054, -0.122])
    create_obs_action_func(trajectory_data_values, file_dir)
    # create_reward(trajectory_data_values, final_state, file_dir)
    # epi_start = np.array([True])
    # for i in range(trajectory_data_values.shape[0]):
    #     epi_start = np.append(epi_start, False)
    # np.save(file_dir + 'episode_starts', epi_start)
    # check = np.load(file_dir + 'episode_returns.npy')
    # print(check, check.shape)
