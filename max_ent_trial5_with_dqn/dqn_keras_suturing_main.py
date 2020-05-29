from dqn_keras_suturing import Agent
import numpy as np
import numba as nb
import concurrent.futures



class RobotStateUtils(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, grid_size, discount, terminal_state_val_from_trajectory):
        super(RobotStateUtils, self).__init__(max_workers=48)
        # Model here means the 3D cube being created
        # linspace limit values: limit_values_pos = [[-0.009, -0.003], [0.003, 007], [-0.014, -0.008]]
        # Creates the model state space based on the maximum and minimum values of the dataset provided by the user
        # It is for created a 3D cube with 3 values specifying each cube node
        # The value 11 etc decides how sparse the mesh size of the cube would be
        self.grid_size = grid_size
        self.lin_space_limits_x = np.linspace(-0.025, 0.025, self.grid_size, dtype='float32')
        self.lin_space_limits_y = np.linspace(0.03, 0.08, self.grid_size, dtype='float32')
        self.lin_space_limits_z = np.linspace(-0.14, -0.09, self.grid_size, dtype='float32')

        # Creates a dictionary for storing the state values
        self.states = {}
        # Creates a dictionary for storing the action values
        self.action_space = {}
        # Numerical values assigned to each action in the dictionary
        # Total Number of states defining the state of the robot
        self.n_params_for_state = 3
        # The terminal state value which is taken from the expert trajectory data
        self.terminal_state_val = terminal_state_val_from_trajectory
        # Deterministic or stochastic transition environment
        self.trans_prob = 1
        # Initialize number of states and actions in the state space model created
        self.n_states = grid_size ** 3
        self.n_actions = 27
        self.gamma = discount

        self.rewards = []
        self.values_tmp = np.zeros([self.n_states])

    def create_state_space_model_func(self):
        # Creates the state space of the robot based on the values initialized for linspace by the user
        # print "Creating State space "
        state_set = []
        for i_val in self.lin_space_limits_x:
            for j_val in self.lin_space_limits_y:
                for k_val in self.lin_space_limits_z:
                    # Rounding state values so that the values of the model, dont take in too many floating points
                    state_set.append([round(i_val, 4), round(j_val, 4), round(k_val, 4)])
        # Assigning the dictionary keys
        for i in range(len(state_set)):
            state_dict = {i: state_set[i]}
            self.states.update(state_dict)

        return self.states

    def create_action_set_func(self):
        # Creates the action space required for the robot. It is defined by the user beforehand itself
        action_set = []
        for pos_x in [-0.005, 0, 0.005]:
            for pos_y in [-0.005, 0, 0.005]:
                for pos_z in [-0.005, 0, 0.005]:
                    action_set.append([pos_x, pos_y, pos_z])
        # Assigning the dictionary keys
        for i in range(len(action_set)):
            action_dict = {i: action_set[i]}
            self.action_space.update(action_dict)

        return self.action_space

    def get_state_val_index(self, state_val):
        index_val = abs((state_val[0] + 0.025) / 0.005 * pow(self.grid_size, 2)) + \
                    abs((state_val[1] - 0.03) / 0.005 * pow(self.grid_size, 1)) + \
                    abs((state_val[2] + 0.14) / 0.005)
        return int(round(index_val))

    def is_terminal_state(self, state):

        # because terminal state is being given in array value and needs to convert to index value
        terminal_state_val_index = self.get_state_val_index(self.terminal_state_val)
        if int(state) == int(terminal_state_val_index):
            reward = 10
            # If terminal state is being given as a list then if state == self.terminal_state_val:
            # print "You have reached the terminal state "
            return reward, True
        else:
            reward = rewards[int(state)]
            # It has not yet reached the terminal state
            return reward, False

    def off_grid_move(self, new_state, old_state):

        # Checks if the new state exists in the state space
        if new_state not in self.states.values():
            return True
        # if trying to wrap around the grid, also the reason for the for x in _ is because old_state is a list
        elif (x % self.grid_size for x in old_state) == 0 and (y % self.grid_size for y in
                                                               new_state) == self.grid_size - 1:
            return True
        elif (x % self.grid_size for x in old_state) == self.grid_size - 1 and (y % self.grid_size for y in
                                                                                new_state) == 0:
            return True
        else:
            # If there are no issues with the new state value then return false, negation is present on the other end
            return False


    '''
    def reward_func(self, end_pos_x, end_pos_y, end_pos_z, alpha):
        # Creates list of all the features being considered

        # reward = -1
        if self.is_terminal_state([end_pos_x, end_pos_y, end_pos_z]):
            reward = 0
        else:
            reward = -1

        return reward, 1, 2


    # Created feature set1 which basically takes the exponential of sum of individually squared value
    def features_array_prim_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_1 = np.exp(-(end_pos_x**2))
        return feature_1

    # Created feature set2 which basically takes the exponential of sum of individually squared value
    def features_array_sec_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_2 = np.exp(-(end_pos_y**2))
        # print f2
        return feature_2

    # Created feature set3 which basically takes the exponential of sum of individually squared value
    def features_array_tert_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_3 = np.exp(-(end_pos_z**2))
        return feature_3

    def features_array_sum_func(self, end_pos_x, end_pos_y, end_pos_z):
        feature_4 = np.exp(-(end_pos_x**2 + end_pos_y**2 + end_pos_z**2))
        return feature_4
    '''

    def reset(self):
        self.pos = np.random.randint(0, len(self.states))
        return self.pos

    def step(self, curr_state, action):
        resulting_state = []
        # print "current state", self.states[curr_state]
        # print "action taken", action, self.action_space[action]
        # Finds the resulting state when the action is taken at curr_state
        for i in range(0, self.n_params_for_state):
            resulting_state.append(round(self.states[curr_state][i] + self.action_space[action][i], 4))

        # print "resulting state is ", resulting_state
        # Calculates the reward and returns the reward value, features value and
        # number of features based on the features provided
        # print "reward is ", reward
        # Checks if the resulting state is moving it out of the grid
        resulting_state_index = self.get_state_val_index(resulting_state)
        if not self.off_grid_move(resulting_state, self.states[curr_state]):
            rew, done = self.is_terminal_state(resulting_state_index)
            return resulting_state, rew, done, None
        else:
            # If movement is out of the grid then just return the current state value itself
            # print "*****The value is moving out of the grid in step function*******"
            return self.states[curr_state], -1, self.is_terminal_state(curr_state), None

    def action_space_sample(self):
        # print "random action choice ", np.random.randint(0, len(self.action_space))
        return np.random.randint(0, len(self.action_space))

    # def features_func(self, end_pos_x, end_pos_y, end_pos_z):
    #
    #     features = [self.features_array_prim_func, self.features_array_sec_func, self.features_array_tert_func]
    #     features_arr = []
    #     for n in range(0, len(features)):
    #         features_arr.append(features[n](end_pos_x, end_pos_y, end_pos_z))
    #     # Created the feature function assuming everything has importance, so therefore added each parameter value
    #     return features_arr
    '''
    def get_transition_states_and_probs(self, curr_state, action):

        if self.is_terminal_state(curr_state):
            return [(curr_state, 1)]
        resulting_state = []
        if self.trans_prob == 1:
            for i in range(0, self.n_params_for_state):
                resulting_state.append(round(self.states[curr_state][i] + self.action_space[action][i], 4))
            resulting_state_index = self.get_state_val_index(resulting_state)

            if not self.off_grid_move(resulting_state, self.states[curr_state]):
                # return resulting_state, reward, self.is_terminal_state(resulting_state_index), None
                return [(resulting_state_index, 1)]
            else:
                # if the state is invalid, stay in the current state
                return [(curr_state, 1)]

    def get_transition_mat_deterministic(self):

        self.n_actions = len(self.action_space)
        for si in range(self.n_states):
            for a in range(self.n_actions):
                probabilities = self.get_transition_states_and_probs(si, a)

                for next_pos, prob in probabilities:
                    # sj = self.get_state_val_index(posj)
                    sj = int(next_pos)
                    # Prob of si to sj given action a
                    prob = int(prob)
                    self.P_a[si, a, sj] = prob
        return self.P_a

    def calc_value_for_state(self, s):
        value = max([sum(
            [self.P_a[s, a, s1] * (self.rewards[s] + self.gamma * self.values_tmp[s1]) for s1 in range(self.n_states)])
                     for a in range(self.n_actions)])
        return value, s

    def value_iteration(self, rewards, error=1):
        # Initialize the value function

        values = np.zeros([self.n_states])
        states_range_value = range(0, self.n_states)
        # print "states range value is ", states_range_value
        self.rewards = rewards
        # estimate values
        while True:
            # Temporary copy to check find the difference between new value function calculated & current value function
            # to ensure improvement in value
            self.values_tmp = values.copy()
            # t_value = TicToc()
            # t_value.tic()
            for q, s in self.map(self.calc_value_for_state, states_range_value):
                values[s] = q
                # print "\nvalues is ", values[s]
            # for s in range(self.n_states):
            #     values[s] = max(
            #         [sum([P_a[s, a, s1] * (rewards[s] + self.gamma * values_tmp[s1])
            #               for s1 in range(self.n_states)])
            #          for a in range(self.n_actions)])

            # t_value.toc('Value function section took')
            # print "values ", values[s]
            if max([abs(values[s] - self.values_tmp[s]) for s in range(self.n_states)]) < error:
                break
        # generate deterministic policy
        policy = np.zeros([self.n_states])
        for s in range(self.n_states):
            policy[s] = np.argmax([sum([self.P_a[s, a, s1] * (self.rewards[s] + self.gamma * values[s1])
                                        for s1 in range(self.n_states)])
                                   for a in range(self.n_actions)])

        return policy

    def compute_state_visitation_frequency(self, trajectories, optimal_policy):
        n_trajectories = len(trajectories)
        total_states = len(trajectories[0])
        d_states = len(trajectories[0][0])
        T = total_states
        # mu[s, t] is the prob of visiting state s at time t
        mu = np.zeros([self.n_states, T])
        # print "mu is ", mu
        for trajectory in trajectories:
            # print "trajectory is ", trajectory
            # To get the values of the trajectory in the state space created
            trajectory_index = self.get_state_val_index(trajectory[0])
            # int is added because the index returned is float and the index value for array has to be integer
            mu[int(trajectory_index), 0] += 1
        mu[:, 0] = mu[:, 0] / n_trajectories

        for s in range(self.n_states):
            for t in range(T - 1):
                # Computes the mu value for each state once the optimal action is taken
                mu[s, t + 1] = sum([mu[pre_s, t] * self.P_a[pre_s, int(optimal_policy[pre_s]), s]
                                    for pre_s in range(self.n_states)])
        p = np.sum(mu, 1)
        return p
    '''

'''
def max_action(Q, state_val, action_values):
    # print "max action action val ", action_values
    q_values = np.array([Q[state_val, a] for a in action_values])
    # print "values in max action is ", q_values
    action = np.argmax(q_values)
    # print "---max action function action ", action
    # print "max q value ", q_values[action]
    return action_values[action]
'''

rewards = np.array(
    [0.97444, 0.68079, 0.91048, 0.52778, 0.19388, 0.41015, 0.99297, 0.14661, 0.17545, 0.09261, 0.37618, 0.86123,
     0.43476, 0.12285, 3.19177, 2.06648, 0.31043, 0.11092, 0.24442, 0.58259, 0.91080, 0.91745, 0.62474, 0.13303,
     1.78467, 1.56376, 0.89602, 0.56485, 0.55988, 0.19699, 0.78184, 0.78839, 0.30588, 0.29636, 2.10013, 2.38590,
     0.63988, 0.19145, 0.46986, 0.71966, 0.10489, 0.24274, 0.39292, 0.69782, 0.82711, 0.66796, 2.14736, 0.42038,
     0.66297, 0.95124, 0.95503, 0.34716, 0.49630, 0.31748, 0.56510, 0.87197, 0.95401, 0.73150, 0.88848, 0.95003,
     0.96776, 0.46761, 0.63241, 0.14408, 0.87706, 0.51201, 0.47399, 0.77789, 0.05308, 0.35303, 0.39253, 0.11196,
     0.21113, 0.12664, 0.46087, 0.76664, 0.96691, 0.45228, 0.47102, 0.33073, 0.91178, 0.06961, 0.29378, 0.39724,
     0.24888, 0.71564, 0.07273, 0.65613, 0.10489, 0.46578, 0.22513, 0.21622, 0.35415, 0.63220, 0.17049, 0.21064,
     0.78732, 0.89097, 0.33545, 0.49994, 0.44958, 0.65754, 0.54356, 0.22613, 0.86313, 0.66162, 0.30893, 0.95307,
     0.33894, 0.56673, 0.42397, 0.40154, 0.72859, 0.04483, 0.69268, 0.68078, 0.78674, 0.65911, 0.80561, 0.05066,
     0.01038, 0.92374, 0.80091, 0.20735, 0.52347, 0.50181, 0.62061, 0.40001, 0.21961, 0.65686, 0.92367, 0.76922,
     0.79335, 0.96784, 1.89029, 4.82433, 1.64329, 0.77091, 0.77672, 0.46237, 0.25211, 0.58025, 0.76942, 0.19495,
     2.94822, 3.63364, 1.75803, 0.45832, 0.01748, 0.85659, 0.32123, 0.75670, 0.35895, 0.06883, 0.58853, 6.44981,
     4.49501, 2.27494, 0.68508, 0.70234, 0.74689, 0.29828, 0.11077, 0.66629, 0.63597, 0.15124, 1.71944, 1.63701,
     0.42209, 0.26829, 0.82455, 0.09098, 0.63476, 0.88460, 0.28084, 0.59271, 2.29126, 1.84810, 0.87907, 0.35864,
     0.15832, 0.84306, 0.08497, 0.34029, 0.08846, 0.78873, 0.02001, 0.92838, 0.98804, 0.07815, 0.70207, 0.25155,
     0.87642, 0.57776, 0.28543, 0.76426, 0.89502, 0.65760, 0.59578, 0.75720, 0.23199, 0.61011, 0.02649, 0.86097,
     0.58124, 0.11274, 0.01373, 0.20203, 0.34865, 0.38305, 0.39274, 0.19203, 0.85074, 0.51456, 0.44595, 0.47104,
     0.49301, 0.10979, 0.17642, 0.69908, 0.08254, 0.26678, 0.97593, 0.77352, 0.34247, 0.15631, 0.42176, 0.76902,
     0.98704, 0.23275, 0.53116, 0.80220, 0.45426, 0.15051, 0.69367, 0.36004, 0.49095, 0.94444, 0.37350, 0.85931,
     0.10096, 0.31171, 0.95758, 0.36469, 0.92246, 0.46181, 0.52694, 0.00446, 0.37469, 0.11931, 0.28830, 0.78130,
     0.22553, 0.82232, 0.12388, 2.15739, 1.93933, 0.10551, 0.78326, 0.20543, 0.03381, 0.95262, 0.58953, 0.43972,
     0.94996, 2.38691, 3.52054, 2.39629, 0.49071, 0.05389, 0.32126, 0.60539, 0.95932, 0.09341, 0.06206, 0.32720,
     1.53406, 3.16672, 0.62834, 0.27216, 0.04700, 0.49540, 0.87291, 0.67827, 0.33700, 0.36029, 0.76807, 0.97126,
     2.20806, 0.75369, 0.25478, 0.40146, 0.60744, 0.46388, 0.94047, 0.52800, 0.68169, 2.04681, 0.59650, 0.07725,
     0.03522, 0.15714, 0.84063, 0.01380, 0.85741, 0.42815, 0.56687, 0.95338, 3.11698, 0.78571, 0.90266, 0.69168,
     0.62493, 0.06389, 0.30040, 0.53793, 0.23427, 0.19733, 0.21465, 0.79609, 0.87392, 0.21518, 0.67827, 0.06536,
     0.05388, 0.19102, 0.41722, 0.82331, 0.55517, 0.50283, 0.21068, 0.28455, 0.27044, 0.47149, 0.73235, 0.53667,
     0.19229, 0.14486, 0.91532, 0.78667, 0.90734, 0.25391, 0.28314, 0.30157, 0.89005, 0.09766, 0.83034, 0.69061,
     0.63293, 0.95730, 0.57515, 0.93641, 0.68503, 0.65418, 0.62313, 0.81290, 0.43579, 0.92363, 0.27445, 0.00284,
     0.27186, 0.79764, 0.05262, 0.76797, 0.86008, 2.04297, 1.81235, 0.21237, 0.08316, 0.74776, 0.28096, 0.12705,
     0.72588, 0.58571, 0.00313, 0.67590, 2.04131, 1.72905, 2.01005, 0.32909, 0.08839, 0.51635, 0.01423, 0.56280,
     0.11176, 0.52594, 0.79006, 1.59860, 0.60932, 0.41745, 0.58824, 0.21320, 0.86214, 0.63779, 0.37087, 0.25403,
     0.94852, 0.67942, 3.82434, 0.30171, 0.82775, 0.47379, 0.54063, 0.60508, 0.97890, 0.86751, 0.90668, 0.25393,
     0.00928, 2.00921, 0.82387, 0.19344, 0.65102, 0.99831, 0.07806, 0.58644, 0.95355, 0.42049, 0.03548, 1.59920,
     2.19626, 0.03415, 0.44548, 0.39393, -0.10465, 0.61135, 0.48755, 0.50124, 0.44106, 1.54760, 0.30054, 0.55247,
     0.39558, 0.42006, 0.25281, 0.10824, 0.07567, 0.26224, 0.81612, 0.72364, 0.74339, 0.42157, 0.87457, 0.26737,
     0.73416, 0.24346, 0.52853, 0.16763, 0.82885, 0.93117, 0.68449, 0.29026, 0.26847, 0.26310, 0.33431, 0.54380,
     0.82395, 0.83926, 0.37107, 0.30196, 0.11107, 0.34465, 0.34062, 0.02965, 0.70607, 0.71949, 0.11447, 0.73837,
     0.10763, 0.62211, 0.39527, 0.93149, 0.08034, 0.19841, 0.32348, 0.89278, 0.84949, 0.20333, 0.23009, 0.15330,
     0.17729, 0.17657, 0.88848, 0.72367, 0.48446, 0.07265, 0.30254, 2.97814, 2.15103, 0.00459, 0.70493, 0.39177,
     0.54826, 0.39387, 0.14654, 0.84889, 0.93945, 0.40309, 2.26310, 3.69426, 0.01054, 0.84597, 0.04158, 0.82474,
     0.99692, 0.06590, 0.38029, 0.89413, 0.05607, 0.74266, 1.46484, 0.74942, 0.15989, 0.40809, 0.43605, 0.62118,
     0.99851, 0.69755, 1.85759, 3.68516, 0.56598, 0.96113, 0.86519, 0.38891, 0.00036, 0.62398, 0.44956, 0.14976,
     1.56817, 2.90361, 1.52459, 1.76204, 0.29614, 0.06454, 0.22756, 0.13486, 0.88698, 0.73222, 0.57826, 0.59251,
     0.77417, 0.86124, 0.72174, 0.26363, 0.13629, -0.12664, 0.48440, 0.45248, 0.38886, 0.11376, 0.37301, 0.41101,
     0.84507, 2.39869, 3.01833, 1.55188, 0.45947, 0.24157, 0.02710, 0.06526, 0.61971, 0.56725, 0.66971, 0.67606,
     0.12018, 0.65914, 0.65383, 0.46530, 0.18548, 0.20749, 0.46511, 0.97402, 0.18332, 0.35162, 0.59800, 0.00811,
     0.56941, -0.52862, 0.46250, 0.35766, 0.37530, -0.50386, 0.16739, 0.10747, 0.09189, 0.44692, 0.48699, 0.76823,
     0.59222, 0.00607, 0.32077, 0.73477, 0.85062, 0.40879, 0.73486, 0.50095, 0.04970, 0.50864, 0.30797, 0.58033,
     0.58207, 0.46780, 0.07293, 0.91640, 0.69403, 0.09036, 0.77981, 0.76906, 2.28711, 0.35886, 0.30112, 0.60695,
     0.67125, 0.05178, 0.70011, 0.62365, 0.48304, 0.10701, 0.66998, 1.65351, 0.35501, 0.74006, 0.05871, 0.84057,
     0.70128, 0.01600, 0.85142, 0.23977, 0.08168, 0.55700, 0.78789, 2.26372, 0.98417, 0.79020, 0.23474, 0.26189,
     0.40767, 0.86555, 0.74086, 0.63499, 0.53034, 0.77260, 1.74092, 0.37010, 0.18997, 0.13191, 0.06530, 0.18729,
     0.58292, 0.67845, 0.90889, 0.80030, 1.60316, 2.29524, 0.64832, 0.67573, 0.91768, 0.31245, 0.32149, 0.80811,
     0.46005, 0.87893, 0.33819, 1.88182, 2.09706, 0.46620, 0.36990, 0.13522, 0.05744, 0.27199, 0.75443, 0.36542,
     0.63133, 0.71896, 1.55212, 3.46144, 1.62680, -0.56640, 0.29952, 0.13780, 0.09307, 0.64376, 0.34718, 0.43778,
     0.32054, 0.15742, 0.67854, 0.08062, 0.20365, -0.56345, 0.02193, 0.64223, 0.56415, 0.79245, 0.75680, 0.66538,
     0.75646, 0.64819, 0.02787, 0.41004, 0.06634, 0.66469, 0.20423, 0.54846, 0.82181, 0.01706, 0.53491, 0.43751,
     0.95102, 0.76053, 0.95571, 0.47384, 0.49650, 0.08872, 0.03172, 0.22321, 0.22615, 0.17326, 0.96224, 0.67375,
     0.98023, 0.82036, 0.39741, 0.49052, 0.23582, 0.11849, 0.59500, 0.17583, 0.22187, 0.78727, 0.66916, 0.02307,
     0.71387, 0.18649, 0.33489, 0.61822, 0.29481, 0.70732, 0.88019, 0.82547, 1.80877, 0.27292, 0.28385, 0.46850,
     0.66090, 0.20900, 0.34870, 0.79144, 0.45672, 0.62336, 2.07324, 2.33786, 0.60949, 0.63283, 0.35759, 0.52115,
     0.01261, 0.85474, 0.92466, 0.24238, 0.44290, 1.90108, 0.15165, 0.41961, 0.89962, 0.71688, 0.47478, 0.68861,
     0.19111, 0.57497, 0.83115, 0.81513, 0.48113, 0.72825, 2.37565, 0.05992, 0.01577, 0.95273, 0.49463, 0.60940,
     0.63083, 0.24071, 0.23203, 0.80775, 3.04567, 3.32104, 0.19984, 0.42170, 0.05157, 0.12205, 0.99436, 0.45816,
     0.12862, 0.64258, 0.67169, 0.80482, 2.08188, 0.50725, 0.24401, 0.25289, 0.04275, 0.45055, 0.54457, 0.32310,
     0.97563, 0.45918, 0.40878, 0.72866, 0.05346, 0.96013, 0.50704, 0.91770, 0.22900, 0.48715, 0.23810, 0.68163,
     0.10294, 0.63114, 0.01029, 0.83844, 0.13721, 0.59907, 0.44279, 0.18516, 0.95006, 0.22359, 0.42203, 0.13141,
     0.67277, 0.35282, 0.42047, 0.37198, 0.94628, 0.86768, 0.26735, 0.14861, 0.57191, 0.14533, 0.53085, 0.92194,
     0.48001, 0.66988, 0.18742, 0.95544, 0.96227, 0.89081, 0.28613, 0.92246, 0.89871, 0.16062, 0.53077, 0.32597,
     0.80694, 0.94451, 0.38474, 0.73061, 0.88271, 0.63313, 0.37023, 0.43877, 0.10240, 0.13337, 0.07918, 0.43545,
     0.59749, 0.16338, 0.83850, 0.20338, 0.82713, 0.71016, 0.47897, 0.97646, 0.54478, 0.20292, 0.74636, 0.72588,
     0.03537, 0.72127, 0.60051, 0.34191, 0.28488, 0.42195, 2.22499, 0.91908, 0.42535, 0.71840, 0.98082, 0.51940,
     0.04601, 0.96261, 0.29917, 0.74440, 2.32963, 3.00548, 0.97184, 1.45903, 0.71491, 0.79487, 0.01800, 0.97729,
     0.74920, 0.22697, 0.91219, 0.72091, 0.71071, 1.45920, 2.27058, 0.72804, 0.92652, 0.21978, 0.44748, 0.51675,
     0.54847, 0.80562, 0.20593, 0.15189, 0.01424, 0.09947, 0.68454, 0.94629, 0.65106, 0.28929, 0.38934, 0.04051,
     0.55865, 0.17520, 0.86416, 0.39771, 0.40501, 0.01239, 0.68499, 0.80313, 0.20687, 0.13467, 0.96884, 0.09395,
     0.12822, 0.50748, 0.90367, 0.66660, 0.79147, 0.21568, 0.59161, 0.14626, 0.70104, 0.46301, 0.15585, 0.04817,
     0.32665, 0.71003, 0.14277, 0.69061, 0.42076, 0.79598, 0.62811, 0.23862, 0.70657, 0.44685, 0.91533, 0.21344,
     0.93674, 0.15452, 0.07498, 0.03917, 0.57018, 0.05414, 0.98634, 0.28934, 0.88165, 0.37260, 0.90376, 0.91079,
     0.94610, 0.76213, 0.66173, 0.25764, 0.74826, 0.91521, 0.11810, 0.72686, 0.88699, 0.98604, 0.02007, 0.52027,
     0.76976, 0.91286, 0.76706, 0.02601, 0.01693, 0.94678, 0.15862, 0.31699, 0.50467, 0.71274, 0.69220, 0.11631,
     0.45399, 0.37566, 0.44675, 0.99972, 0.57649, 0.70729, 0.79630, 0.17518, 0.97113, 0.86633, 0.80923, 0.12177,
     0.05155, 0.56064, 0.78701, 0.83448, 0.90246, 0.56990, 0.86047, 0.84716, 0.71480, 0.09224, 0.53348, 0.55977,
     0.05868, 0.28121, 0.68527, 0.32299, 0.33076, 0.23471, 0.98099, 0.53434, 0.43460, 0.87431, 0.57646, 0.44372,
     0.43633, 0.00914, 0.07966, 0.17722, 0.40968, 0.05851, 0.82750, 0.91900, 0.63278, 0.45385, 0.17688, 0.29818,
     0.26790, 0.45250, 0.05376, 0.16792, 0.18194, 0.46969, 0.83426, 0.44448, 0.60624, 0.64137, 0.90661, 0.45546,
     0.57854, 0.45105, 0.66736, 0.84815, 0.11335, 0.03963, 0.99859, 0.70842, 0.74358, 0.74481, 0.28768, 0.45623,
     0.11676, 0.72560, 0.32438, 0.94559, 0.80295, 0.36047, 0.15732, 0.54555, 0.41195, 0.84039, 0.04795, 0.11228,
     0.61843, 0.60410, 0.19136, 0.71825, 0.33384, 0.61896, 0.45063, 0.51114, 0.44464, 0.78993, 0.23603, 0.06939,
     0.77644, 0.63411, 0.38137, 0.94716, 0.12725, 0.58603, 0.97774, 0.54197, 0.75979, 0.68714, 0.07203, 0.05113,
     0.23857, 0.98455, 0.26658, 0.03786, 0.14784, 0.63307, 0.30138, 0.67907, 0.78351, 0.99729, 0.08464, 0.65877,
     0.50918, 0.71810, 0.15197, 0.26035, 0.02133, 0.94128, 0.26591, 0.94303, 0.18205, 0.95896, 0.10326, 0.19976,
     0.47107, 0.70772, 0.33959, 0.27142, 0.81133, 0.86438, 0.32051, 0.38999, 0.01758, 0.20160, 0.35235, 0.91352,
     0.76449, 0.67412, 0.18436, 0.59994, 0.00180, 0.01875, 0.72016, 0.51174, 0.34221, 0.85375, 0.23498, 0.94883,
     0.70373, 0.48914, 0.41017, 0.92225, 0.35687, 0.22270, 0.28148, 0.22158, 0.81303, 0.84361, 0.87916, 0.08003,
     0.35876, 0.11279, 0.73352, 0.37825, 0.80114, 0.69102, 0.88144, 0.96743, 0.03890, 0.59764, 0.48598, 0.77626,
     0.66179, 0.93408, 0.87176, 0.57577, 0.71702, 0.59194, 0.68752, 0.02305, 0.62395, 0.07258, 0.62533, 0.41076,
     0.54218, 0.14320, 0.75441, 0.75644, 0.95444, 0.48090, 0.07898, 0.70862, 0.50746, 0.35946, 0.42699, 0.03960,
     0.18743, 0.49497, 0.70733, 0.40817, 0.88927, 0.44123, 0.84993, 0.18793, 0.68734, 0.06442, 0.17154, 0.34037,
     0.29526, 0.89693, 0.00660, 0.93264, 0.53192, 0.33920, 0.41454, 0.89487, 0.25376, 0.43518, 0.23881, 0.16467,
     0.53073, 0.07903, 0.06949, 0.84708, 0.12688, 0.46644, 0.06987, 0.30856, 0.80237, 0.56071, 0.01167, 0.19458,
     0.22094, 0.17237, 0.65191, 0.93568, 0.75293, 0.90432, 0.87916, 0.78175, 0.20364, 0.08256, 0.28974, 0.15960,
     0.13697, 0.64159, 0.72875, 0.18746, 0.68807, 0.10072, 0.22569, 0.55207, 0.98617, 0.48424, 0.35746, 0.47711,
     0.78719, 0.40782, 0.53436, 0.88648, 0.86031, 0.49046, 0.03026, 0.56241, 0.91220, 0.97289, 0.19137, 0.99420,
     0.47252, 0.59805, 0.34327, 0.16082, 0.64182, 0.09286, 0.28195, 0.45778, 0.97528, 0.02457, 0.41278, 0.22545,
     0.98329, 0.95770, 0.55309, 0.55460, 0.54867, 0.68086, 0.66431, 0.18649, 0.50904, 0.12220, 0.26274, 0.77611,
     0.69548, 0.42966, 0.83722, 0.36158, 0.19291, 0.23935, 0.34716, 0.90948, 0.95442, 0.21818, 0.64397, 0.73033,
     0.83759, 0.96149, 0.49584, 0.57830, 0.38767, 0.75918, 0.36259, 0.49428, 0.82487, 0.01940, 0.58134, 0.81556,
     0.81083, 0.73072, 0.81453, 0.40833, 0.43071, 0.32831, 0.01615, 0.49016, 0.22278, 0.99521, 0.96482])


def dqn_model(env_obj, alpha, gamma, epsilon):
    lr = 0.0005
    load_checkpoint = False
    n_episodes = 500
    agent = Agent(gamma=gamma, epsilon=epsilon, alpha=alpha, input_dims=3,
                  n_actions=27, mem_size=1000000, batch_size=32, epsilon_end=0.0)
    if load_checkpoint:
        agent.load_model()
    scores = []
    eps_history = []

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env_obj.reset()
        while not done:
            action = agent.choose_action(env_obj.states[observation])
            # print "action chosen ", action
            observation_, reward, done, info = env_obj.step(env_obj.states[observation], action)
            # print "resulting state is ", observation_
            score += reward
            agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score: %.2f' % score,
              ' average score %.2f' % avg_score)

        if i % 10 == 0 and i > 0:
            agent.save_model()


if __name__ == '__main__':
    goal = np.array([-0.005, 0.06, -0.125])
    # term_state = np.random.randint(0, grid_size ** 3)]
    # Pass the required gridsize, discount, terminal_state_val_from_trajectory):
    env_obj = RobotStateUtils(11, 0.9, goal)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    print "State space created is ", states
    policy = np.zeros(len(states))
    print "states is ", states[553]
    print "actions are ", action

    dqn_model(env_obj, alpha=0.1, gamma=0.99, epsilon=1.0)

    # x = [i+1 for i in range(n_games)]
    # plotLearning(x, scores, eps_history, filename)
