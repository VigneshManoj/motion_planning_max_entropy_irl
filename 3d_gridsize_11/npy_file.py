import numpy as np
# trajectories = np.load('/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/trajectory_class_1.npz')
# state_trajectories = trajectories['state']
# action_trajectories = trajectories['action']
# print state_trajectories
# for state_trajectory in state_trajectories:
#     # print "state traj is ", state_trajectory
#     for iter in range(0, state_trajectory.shape[0]):
#         x = np.atleast_2d(state_trajectory[iter, 0])
#         # print "x val is ", x
trajectories  = np.load('/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/trajectories_data.npz')
print trajectories['state']
print trajectories['action']