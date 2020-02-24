import numpy as np
from numpy import savetxt

from robot_markov_model import RobotMarkovModel
from robot_state_utils import RobotStateUtils


class MaxEntIRL:

    def __init__(self, trajectory_length, grid_size):
        # super(MaxEntIRL, self).__init__(11)
        self.trajectory_length = trajectory_length
        self.grid_size = grid_size

    # Calculates the reward function weights using the Max Entropy Algorithm
    def max_ent_irl(self, trajectories, discount, n_trajectories, epochs, learning_rate, filename):
        # Finds the total number of states and dimensions of the list of features array
        # 0 indicates that the first trajectory data is being used
        total_states = len(trajectories[0])
        # total_states indicates that the last value of that trajectory data should be used as terminal state
        terminal_state_val_from_trajectory = trajectories[0][total_states-1]
        print "Terminal state value is ", terminal_state_val_from_trajectory
        # Creates the environment object created for the robot workspace
        env_obj = RobotStateUtils(self.grid_size, discount, terminal_state_val_from_trajectory)
        # Creates the states and actions for environment
        states = env_obj.create_state_space_model_func()
        action = env_obj.create_action_set_func()
        print "states is ", states
        print "action are ", action
        # Creates the feature matrix for the given robot environment
        # Currently a identity feature matrix is being used
        feat_map = np.eye(self.grid_size ** 2)
        # feat_map = env_obj.get_feature_matrix()
        # Initialize with random weights based on the dimensionality of the states
        weights = np.random.uniform(size=(feat_map.shape[1],))
        # Find feature expectations, sum of features of trajectory/number of trajectories
        feature_expectations = self.find_feature_expectations(feat_map, trajectories)
        # print "feature array is ", feature_array_all_trajectories[0][0:total_states]
        # Gradient descent on alpha
        for i in range(epochs):
            if i % 25 == 0:
                print "Epoch running is ", i
                print "weights is ", weights
                file_open = open(filename, 'a')
                savetxt(filename, weights, delimiter=',', fmt="%10.5f", newline=", ")
                file_open.write("\n \n \n \n ")
                file_open.close()
            reward = np.dot(feat_map, weights)
            optimal_policy, expected_svf = self.find_expected_svf(weights, discount, total_states, trajectories, reward)
            # Computes the gradient
            grad = feature_expectations - feat_map.T.dot(expected_svf)
            # Change the learning rate based on the iteration running
            if i < 50:
                learning_rate = 0.1
            else:
                learning_rate = 0.01
            # Gradient descent for finding new value of weights
            weights += learning_rate * np.transpose(grad)
            # print "weights is ", weights
        # Compute the reward of the trajectory based on the weights value calculated
        trajectory_reward = np.dot(feat_map, weights)

        return trajectory_reward, weights

    def find_feature_expectations(self, feat_map, trajectories):
        # Initialization of feat_exp
        feat_exp = np.zeros([feat_map.shape[1]])
        # For each trajectory in the array of all trajectories
        for trajectory in trajectories:
            # Each state value for the trajectory followed
            for state_value in trajectory:
                # print "state value is ", state_value
                # Calculates the index value for the given state value
                state_value_index = self.get_state_val_index(state_value)
                # print "feat map is ", feat_map[state_value_index, :]
                # The feat exp is updated based on the states being visited in the expert data
                feat_exp += feat_map[state_value_index, :]
        # Divide by the number of trajectories to get the average
        feat_exp = feat_exp / len(trajectories)

        return feat_exp

    def find_expected_svf(self, weights, discount, total_states, trajectories, reward):
        # Creates object of robot markov model
        robot_mdp = RobotMarkovModel()
        # Returns the state and action values of the state space being created
        state_values_from_trajectory, _ = robot_mdp.return_trajectories_data()
        # Finds the terminal state value from the expert trajectory
        # 0 indicates that the first trajectory data is being used
        # total_states indicates that the last value of that trajectory data should be used as terminal state
        terminal_state_val_from_trajectory = state_values_from_trajectory[0][total_states-1]
        # Pass the gridsize required
        # To create the state space model, we need to run the below commands
        env_obj = RobotStateUtils(self.grid_size, discount, terminal_state_val_from_trajectory)
        states = env_obj.create_state_space_model_func()
        action = env_obj.create_action_set_func()
        # print "State space created is ", states
        P_a = env_obj.get_transition_mat_deterministic()
        # print "P_a is ", P_a
        # print "rewards is ", rewards
        policy = env_obj.value_iteration(reward)
        # policy = np.random.randint(27, size=1331)
        # print "policy is ", policy
        # Finds the sum of features of the expert trajectory and list of all the features of the expert trajectory
        expected_svf = env_obj.compute_state_visitation_frequency(trajectories, policy)
        # print "svf is ", expected_svf


        # Returns the policy, state features of the trajectories considered in the state space and expected svf
        return policy, expected_svf

    def get_state_val_index(self, state_val):
        index_val = abs((state_val[0] + 5) * pow(self.grid_size, 1)) + \
                    abs((state_val[1] + 5) * pow(self.grid_size, 0))
        # Returns the index value
        return int(index_val)







