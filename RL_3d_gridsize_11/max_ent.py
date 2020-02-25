import numpy as np
import math
import features
# import matplotlib.pyplot as plt
from util_func import RobotMarkovModel
import sys
import pandas as pd
# n_iterations = int(sys.argv[1])
# rl_iter = int(sys.argv[2])
# svf_iter = int(sys.argv[3])
n_iterations = 1
rl_iter = 2
svf_iter = 2
print rl_iter, svf_iter
# Using csv file to read the data and find states and actions
trajectories = np.genfromtxt ("/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/sample_trajectory_data_without_norm.csv", delimiter=",")
state_trajectories = trajectories[:, 0:6]
action_trajectories = trajectories[:, 6:12]
print "type is", type(state_trajectories)
# If planning to read from npz file use the following, it gives same output as above
# trajectories  = np.load('/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/trajectories_data.npz')
# print trajectories['state']
# print trajectories['action']
mdp_obj = RobotMarkovModel()
n_traj = len(state_trajectories)

weights = np.random.rand(1, 2)
print weights
Z = np.empty([0, 1])
trajectories_probability = np.empty([len(state_trajectories), 1], dtype='float32')
for n in range(0, n_iterations):
    print "Iteration: ", n
    trajectories_reward = []
    trajectories_features = []
    trajectory_reward = np.zeros([1, 1], dtype='float32')
    trajectory_features = np.zeros([2, 1], dtype='float32')
    for iter in range(0, state_trajectories.shape[0]):
        rot_par_r = state_trajectories[iter, 0]
        rot_par_p = state_trajectories[iter, 1]
        rot_par_y = state_trajectories[iter, 2]
        end_pos_x = state_trajectories[iter, 3]
        end_pos_y = state_trajectories[iter, 4]
        end_pos_z = state_trajectories[iter, 5]

        r, f = features.reward(np.array([rot_par_r, rot_par_p, rot_par_y, end_pos_x, end_pos_y, end_pos_z]), weights)
        trajectory_reward = trajectory_reward + r
        trajectory_features = trajectory_features + np.vstack((f[0], f[1]))
    trajectories_reward.append(trajectory_reward)
    trajectories_features.append(trajectory_features)
    # print trajectory_features
    # print len(trajectories_reward)
    trajectories_probability = np.exp(trajectories_reward)
    feature_state, policy = mdp_obj.get_policy(weights, rl_iter, svf_iter)
    # print sum(feature_state.reshape(301*301*101*11,1))
    Z = np.vstack((Z, sum(trajectories_reward)))
    # # trajectories_probability.reshape((len(trajectories_reward),1))
    # L=np.vstack((L,sum(trajectories_reward)/n_traj - np.log(Z)))
    # # if L[n]<L[n-1]:
    # #     break
    #
    grad_L = sum(trajectories_features)/n_traj - feature_state.reshape(2, 1)
    print grad_L.shape
    #
    weights = weights + 0.005*np.transpose(grad_L)
    print Z[n]
# np.save('final_policy', policy)
# np.save('final_weights', weights)
print "Weights are:", weights
# print "Likelihood is :", L
# fig = plt.figure()
# ax = fig.gca()
# ax.plot(Z)
# plt.show()
