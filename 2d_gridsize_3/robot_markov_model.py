import numpy as np


class RobotMarkovModel:
    def __init__(self):
        # Reads the trajectory data from the file
        trajectories1 = np.genfromtxt\
            ("/home/vignesh/PycharmProjects/iros2020/data/check_data_max_ent_trial5_code1.csv",
             delimiter=",")
        trajectories2 = np.genfromtxt\
            ("/home/vignesh/PycharmProjects/iros2020/data/check_data_max_ent_trial5_code2.csv",
             delimiter=",")
        trajectories3 = np.genfromtxt\
            ("/home/vignesh/PycharmProjects/iros2020/data/check_data_max_ent_trial5_code3.csv",
             delimiter=",")
        # Separates the state trajectories data and action data
        self.state_trajectories = []
        self.state_trajectories.append(trajectories1[:, 0:2])
        self.state_trajectories.append(trajectories2[:, 0:2])
        self.state_trajectories.append(trajectories3[:, 0:2])

        self.action_trajectories = []
        self.action_trajectories.append(trajectories1[:, 2:4])
        self.action_trajectories.append(trajectories2[:, 2:4])
        self.action_trajectories.append(trajectories3[:, 2:4])

        # Initialize the actions possible
        self.action_set = []
    # Returns the state and action array of expert trajectory
    def return_trajectories_data(self):
        # Return trajectories data if any function requires it outside this class
        return self.state_trajectories, self.action_trajectories

    def generate_trajectories(self):
        # state values. These values are repeatedly added until the length of trajectory
        individual_feature_array = []
        # feature_array_all_trajectories = np.zeros((3, 185, 3))
        feature_array_all_trajectories = []
        for state_trajectory in self.state_trajectories:
            # It is to reset the list to null and start from 185 again
            individual_feature_array = []
            for i in range(0, len(state_trajectory)):
                # Reads only the state trajectory data and assigns the variables value of the first set of state values
                end_pos_x = state_trajectory[i, 0]
                end_pos_y = state_trajectory[i, 1]
                  # Creates an array of each position and individual trajectory
                individual_feature_array.append([end_pos_x, end_pos_y])
            # print "individual feature ", np.array(individual_feature_array)
            # Joins all the trajectories provided by the expert
            feature_array_all_trajectories.append(np.array(individual_feature_array))
        # Returns the array of sum of all trajectory features and returns the array of all the features of a trajectory
        return np.array(feature_array_all_trajectories)

if __name__ == '__main__':
    obj = RobotMarkovModel()
    s, a = obj.return_trajectories_data()
    # print "states is ", s
    # print "actions is ", a
    # print "len state s", len(s[0])
    feat_array = obj.generate_trajectories()
    print "features ", feat_array
    print "len is ", len(feat_array)
    feat_map = np.eye(11 ** 2)
    feat_exp = np.zeros([feat_map.shape[1]])
