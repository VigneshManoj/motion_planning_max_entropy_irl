import pandas as pd
import numpy as np
import csv
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


if __name__ == "__main__":

    file_name = "/home/vignesh/Thesis_Suture_data/trial2/suture_data_trial2/formatplot.csv"
    dir1 = '/home/vignesh/PycharmProjects/motion_planning_max_entropy_irl/RL_3d_gridsize_11/different_policies/'
    states = np.load(dir1 + 'output3.npy')
    states = states.reshape(int(len(states)/3), 3)
    dir2 = '/home/vignesh/PycharmProjects/motion_planning_max_entropy_irl/RL_3d_gridsize_11/random_points_output/'
    states1 = np.load(dir2+'output4.npy')
    states1 = states1.reshape(int(len(states1)/3), 3)
    states2 = np.load(dir2+'output5.npy')
    states2 = states2.reshape(int(len(states2)/3), 3)
    states3 = np.load(dir2+'output6.npy')
    states3 = states3.reshape(int(len(states3)/3), 3)
    states4 = np.load(dir2+'output7.npy')
    states4 = states4.reshape(int(len(states4)/3), 3)
    states5 = np.load(dir2+'output8.npy')
    states5 = states5.reshape(int(len(states5)/3), 3)
    states6 = np.load(dir2+'output9.npy')
    states6 = states6.reshape(int(len(states6)/3), 3)
    states7 = np.load(dir2+'output10.npy')
    states7 = states7.reshape(int(len(states7)/3), 3)
    states8 = np.load(dir2+'output11.npy')
    states8 = states8.reshape(int(len(states8)/3), 3)
    states9 = np.load(dir2+'output12.npy')
    states9 = states9.reshape(int(len(states9)/3), 3)
    # states10 = np.load(dir2+'/random_points_output/output13.npy')
    # states10 = states10.reshape(int(len(states10)/3), 3)

    actual_val = pd.read_csv(file_name).to_numpy()
    # print(states)
    # print(actual_val)
    plt.rcParams.update({'font.size': 12})
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_title("Comparison between Learnt RL Trajectory and Operator Trajectory for Expanded Grid")
    # ax.set_title("Learnt RL Trajectory from different Initial States")
    ax.set_xlabel('X(mm)', labelpad=10)
    ax.set_ylabel('Y(mm)', labelpad=10)
    ax.set_zlabel('Z(mm)', labelpad=10)
    color_rl = 'r'
    color_user = 'g'
    # ax.plot(states10[:, 0], states10[:, 1], states10[:, 2], color=color_rl, linewidth=3)
    # ax.plot(actual_val[:, 0], actual_val[:, 1], actual_val[:, 2], linestyle='--', color=color_user, linewidth=3)
    ax.plot(states[:, 0]*1000, states[:, 1]*1000, states[:, 2]*1000, color=color_rl, linewidth=2)
    ax.plot(actual_val[:, 0]*1000, actual_val[:, 1]*1000, actual_val[:, 2]*1000, linestyle='--', color=color_user, linewidth=2)
    ax.plot((states8[:, 0]+0.005)*1000, (states8[:, 1]+0.05)*1000, (states8[:, 2])*1000, color=color_rl, linewidth=2)
    ax.plot((actual_val[:, 0]+0.005)*1000, (actual_val[:, 1]+0.05)*1000, (actual_val[:, 2])*1000, linestyle='--', color=color_user, linewidth=2)

    # ax.plot(states1[:, 0], states1[:, 1], states1[:, 2], linewidth=3, linestyle=(0, (3, 5, 1, 5, 1, 5)))
    # ax.plot(states4[:, 0], states4[:, 1], states4[:, 2], linewidth=3, linestyle=(0, (1, 10)))
    # ax.plot(states5[:, 0], states5[:, 1], states5[:, 2], linewidth=3, color='m', linestyle=(0, (1, 1)))
    # ax.plot(states7[:, 0], states7[:, 1], states7[:, 2], linewidth=3,  color='y', linestyle=':')
    # ax.plot(states8[:, 0], states8[:, 1], states8[:, 2], linewidth=3,  color='c', linestyle='-.')
    # ax.plot(states9[:, 0], states9[:, 1], states9[:, 2], linewidth=3,  color='r', linestyle=(0, (5, 10)))


    # ax.legend(["From Initial State X, Y, Z =[-0.005, 0.06, -0.135]", "Expert Trajectory"], loc=0, bbox_to_anchor=(0.25, 0.2, 0.5, 0.5))
    # ax.legend(["User's Trajectory", "Initial State 1", "Initial State 2", "Initial State 3", "Initial State 4",
    #           "Initial State 5", "Initial State 6"], loc=0, bbox_to_anchor=(-0.2, 0.65, 0.5, 0.5))
    ax.legend(["RL Learnt Trajectory", "User Trajectory"], loc=0, bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    ax.view_init(45, 135)

    # plt.xticks([-30, -20, -10, 0, 10])
    # plt.yticks([35, 40, 45, 50, 55])

    # ax.set_zticks([-120, -124, -128, -132, -136])
    # plt.savefig('shifted_grid.png')
    # plt.savefig(dir2 + 'expanded_grid.png')
    # plt.savefig(dir2 + '/random_points_output/different_states_output10.png')
    plt.show()


