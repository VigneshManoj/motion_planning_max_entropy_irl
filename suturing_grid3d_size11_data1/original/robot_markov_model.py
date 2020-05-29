import numpy as np


class RobotMarkovModel:
    def __init__(self):
        # Reads the trajectory data from the file
        trajectories1 = np.genfromtxt\
            ("/home/vvarier/dvrk_automated_suturing/iros2020/suturing_grid3d_size11_data1/grid11_parallel/max_ent_grid11_data1.csv",
             delimiter=",")
        # trajectories1 = np.genfromtxt\
        #     ("/home/vvarier/dvrk_automated_suturing/iros2020/RL_3d_gridsize_11/grid11_parallel/max_ent_grid11_data1.csv",
        #      delimiter=",")
        # trajectories3 = np.genfromtxt\
        #     ("/home/vvarier/dvrk_automated_suturing/iros2020/suturing_grid3d_size11/grid11_parallel/max_ent_grid11_data3.csv",
        #      delimiter=",")
        # trajectories1 = np.genfromtxt\
        #    ("/home/vignesh/PycharmProjects/motion_planning_max_entropy_irl/RL_3d_gridsize_11/trial4_grid11_parallel/max_ent_grid11_data1.csv",
        #     delimiter=",")
        # trajectories1 = np.genfromtxt\
        #     ("/home/vignesh/PycharmProjects/dvrk_automated_suturing/data/check_data_max_ent_trial3_code1.csv",
        #      delimiter=",")
        # trajectories2 = np.genfromtxt\
        #     ("/home/vignesh/Thesis_Suture_data/trial2/suture_data_trial2/832953_edited.csv",
        #      delimiter=",")
        # trajectories3 = np.genfromtxt\
        #     ("/home/vignesh/Thesis_Suture_data/trial2/suture_data_trial2/781266_edited.csv",
        #      delimiter=",")
        # Separates the state trajectories data and action data
        self.state_trajectories = []
        self.state_trajectories.append(trajectories1[:, 0:3])
        # self.state_trajectories.append(trajectories2[:, 0:3])
        # self.state_trajectories.append(trajectories3[:, 0:3])

        self.action_trajectories = []
        self.action_trajectories.append(trajectories1[:, 3:6])
        # self.action_trajectories.append(trajectories2[:, 3:6])
        # self.action_trajectories.append(trajectories3[:, 3:6])

        # Initialize the actions possible
        self.action_set = []
    # Returns the state and action array of expert trajectory
    def return_trajectories_data(self):
        # Return trajectories data if any function requires it outside this class
        return self.state_trajectories, self.action_trajectories
    '''
    # Calculates reward function
    def reward_func(self, end_pos_x, end_pos_y, end_pos_z, alpha):
        # Creates list of all the features being considered
        features = [self.features_array_prim_func, self.features_array_sec_func, self.features_array_tert_func]
        reward = 0
        features_arr = []
        for n in range(0, len(features)):
            features_arr.append(features[n](end_pos_x, end_pos_y, end_pos_z))
            # print "alpha size", alpha[0, n].shape
            # print "features size ", features_arr[n].shape
            reward = reward + alpha[0, n]*features_arr[n]
        # Created the feature function assuming everything has importance, so therefore added each parameter value
        return reward, np.array([features_arr]), len(features)

    # Created feature set1 which basically takes the exponential of sum of individually squared value
    def features_array_prim_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_1 = np.exp(-(end_pos_x**2))
        return feature_1

    # Created feature set2 which basically takes the exponential of sum of individually squared value divided by
    # the variance value
    def features_array_sec_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_2 = np.exp(-(end_pos_y**2))
        # print f2
        return feature_2

    def features_array_tert_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_3 = np.exp(-(end_pos_z**2))
        return feature_3

    # It returns the features stacked together for a specific states (depends on how many number of features exist)
    def features_func(self, end_pos_x, end_pos_y, end_pos_z):

        features = [self.features_array_prim_func, self.features_array_sec_func, self.features_array_tert_func]
        features_arr = []
        for n in range(0, len(features)):
            features_arr.append(features[n](end_pos_x, end_pos_y, end_pos_z))
        # Created the feature function assuming everything has importance, so therefore added each parameter value
        return features_arr

    def get_state_val_index(self, state_val):
        index_val = abs((state_val[0] + 0.5) * pow(11, 2)) + abs((state_val[1] + 0.5) * pow(11, 1)) + \
                    abs((state_val[2] + 0.5))
        return round(index_val*(10))
    '''
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
                end_pos_z = state_trajectory[i, 2]
                # Creates an array of each position and individual trajectory
                individual_feature_array.append([end_pos_x, end_pos_y, end_pos_z])
            # print "individual feature ", np.array(individual_feature_array)
            # Joins all the trajectories provided by the expert
            feature_array_all_trajectories.append(np.array(individual_feature_array))
        # Returns the array of sum of all trajectory features and returns the array of all the features of a trajectory
        return np.array(feature_array_all_trajectories)


if __name__ == '__main__':
    obj = RobotMarkovModel()
    s, a = obj.return_trajectories_data()
    print "states is ", s
    # print "len state s", len(s[0])
    sum_feat, feat_array = obj.generate_trajectories()
    # print "sum of features ", sum_feat
    total_states = len(feat_array[0])
    print "states value at 31 is ", s[0][total_states-1]
    print "total states is ", total_states
    print "features array ", len(s[0][0:total_states])
    # print "len ", len(feat_array[0][0])
    d_states = len(feat_array[0][0])
    T = total_states
    # print "total state is ", total_states
    # mu[s, t] is the prob of visiting state s at time t
    mu = np.zeros([1331, T])
    # for trajectory in s:
    #     # print "traj is ", trajectory
    #     ind = obj.get_state_val_index(trajectory[0])
    #     # print "ind is ", ind
    #     mu[int(ind), 0] += 1
    # # print "mu is ", mu

