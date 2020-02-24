# import matplotlib.pyplot as plt
from numpy import savetxt

from robot_markov_model import RobotMarkovModel
from max_ent_irl import MaxEntIRL


def main(grid_size, discount_factor, epochs, learning_rate):
    # Creates an object for using the RobotMarkovModel class
    robot_mdp = RobotMarkovModel()
    # Reads the trajectory data from the csv files provided and returns the trajectories
    # The trajectories is numpy array of the number of trajectories provided
    trajectories = robot_mdp.generate_trajectories()
    # Finds the length of the trajectories data
    n_trajectories = len(trajectories)
    # Initialize the IRL class object, provide trajectory length as input, currently its value is 3
    irl = MaxEntIRL(n_trajectories, grid_size)
    # For storing results
    filename = "/home/vvarier/dvrk_automated_suturing/iros2020/motion_planning_max_entropy_irl/data/weights_grid2d.txt"
    # Calculates the reward function based on the Max Entropy IRL algorithm
    reward, weights = irl.max_ent_irl(trajectories, discount_factor, n_trajectories, epochs, learning_rate, filename)

    print "r is ", reward.reshape((grid_size, grid_size), order='F')
    # print "r shape ", reward.shape
    print "weights is ", weights.reshape((grid_size, grid_size), order='F')
    # print "policy is ", policy[0][0]
    file_open = open(filename, 'a')
    file_open.write("\n \n \n \n Final result \n")
    savetxt(filename, weights, delimiter=',', fmt="%10.5f", newline=", ")
    file_open.close()


if __name__ == '__main__':
    # Epochs indicates the number of times gradient iteration needs to be run
    main(grid_size=51, discount_factor=0.9, epochs=200, learning_rate=0.01)
