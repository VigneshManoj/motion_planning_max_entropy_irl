import numpy as np
import concurrent.futures


class RobotStateUtils(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, grid_size, discount, terminal_state_val_from_trajectory):
        super(RobotStateUtils, self).__init__(max_workers=8)
        # Model here means the 3D cube being created
        # linspace limit values: limit_values_pos = [[-0.009, -0.003], [0.003, 007], [-0.014, -0.008]]
        # Creates the model state space based on the maximum and minimum values of the dataset provided by the user
        # It is for created a 3D cube with 3 values specifying each cube node
        # The value 11 etc decides how sparse the mesh size of the cube would be
        self.grid_size = grid_size
        self.lin_space_limits_x = np.linspace(-0.03, 0.02, self.grid_size, dtype='float32')
        self.lin_space_limits_y = np.linspace(0.025, 0.075, self.grid_size, dtype='float32')
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

        self.rewards = np.zeros([self.n_states])
        # self.P_a = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.int32)
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
        index_val = abs((state_val[0] + 0.03) / 0.005 * pow(self.grid_size, 2)) + \
                    abs((state_val[1] - 0.025) / 0.005 * pow(self.grid_size, 1)) + \
                    abs((state_val[2] + 0.14) / 0.005)
        return int(round(index_val))

    def is_terminal_state(self, state):

        # because terminal state is being given in array value and needs to convert to index value
        terminal_state_val_index = self.get_state_val_index(self.terminal_state_val)
        if int(state) == int(terminal_state_val_index):
            # If terminal state is being given as a list then if state == self.terminal_state_val:
            # print "You have reached the terminal state "
            return True
        else:
            # It has not yet reached the terminal state
            return False

    def off_grid_move(self, new_state, old_state):

        # Checks if the new state exists in the state space
        sum_feat = np.zeros(len(self.states))
        for i, ele in enumerate(self.states.values()):
            sum_feat[i] = np.all(np.equal(ele, new_state))
        if np.sum(sum_feat) == 0:
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

    def reward_func(self, state):
        # Creates list of all the features being considered
        # features = [self.features_array_prim_func, self.features_array_sec_func, self.features_array_tert_func]
        # features_arr = []
        # for n in range(0, len(features)):
        #     features_arr.append(features[n](end_pos_x, end_pos_y, end_pos_z))

        if self.is_terminal_state(state):
            reward = 10
        else:
            reward = 0

        return reward

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
        reward = self.reward_func(resulting_state_index)

        if not self.off_grid_move(resulting_state, self.states[curr_state]):
            return resulting_state, reward, self.is_terminal_state(resulting_state_index), None
        else:
            # If movement is out of the grid then just return the current state value itself
            # print "*****The value is moving out of the grid in step function*******"
            return self.states[curr_state], reward, self.is_terminal_state(curr_state), None

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
                    if self.P_a[si, a, sj] != 1:
                        print "prob is ", self.P_a[si, a, sj]
        return self.P_a

    def calc_value_for_state(self, s):
        value = max([sum(
            [(self.rewards[s] + self.gamma * self.values_tmp[s1]) for s1 in range(self.n_states)])
                     for a in range(self.n_actions)])
        # print "value of state is ", value
        return value, s

    def value_iteration(self, rewards, error=0.01):
        # Initialize the value function

        values = np.zeros([self.n_states])
        states_range_value = range(0, self.n_states)
        # print "states range value is ", states_range_value
        self.rewards = rewards
        # estimate values
        count = 0

        while True:
            # Temporary copy to check find the difference between new value function calculated & current value function
            # to ensure improvement in value
            self.values_tmp = values.copy()
            # t_value = TicToc()
            # t_value.tic()
            for q, s in self.map(self.calc_value_for_state, states_range_value):
                values[s] = q
                count += 1
                if count%250 == 0:
                    print "\nvalues is ", values[s]
            '''
            for s in range(self.n_states):
                values[s] = max(
                    [sum([P_a[s, a, s1] * (rewards[s] + self.gamma * values_tmp[s1])
                          for s1 in range(self.n_states)])
                     for a in range(self.n_actions)])
            '''
            # t_value.toc('Value function section took')
            # print "values ", values[s]
            if max([abs(values[s] - self.values_tmp[s]) for s in range(self.n_states)]) < error:
                break
        # generate deterministic policy
        policy = np.zeros([self.n_states])
        for s in range(self.n_states):
            policy[s] = np.argmax([sum([(self.rewards[s] + self.gamma * values[s1])
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


if __name__ == '__main__':
    goal = np.array([[1, 1, 0]])
    # term_state = np.random.randint(0, grid_size ** 3)]
    env_obj = RobotStateUtils(11, 0.9, goal)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    print "State space created is ", states
    print "actions is ", action
    index_val = env_obj.get_state_val_index(states[53])
    print "index val is ", index_val
    '''
    # Robot Object called
    # Pass the gridsize required
    [0.02, 0.075, -0.09]
    weights = np.array([[1, 1, 0]])
    # term_state = np.random.randint(0, grid_size ** 3)]
    env_obj = RobotStateUtils(11, weights, 0.9, [0.5, 0.5, 0])
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    # print "State space created is ", states
    P_a = env_obj.get_transition_mat_deterministic()
    # print "P_a is ", P_a
    print "shape of P_a ", P_a.shape
    rewards = []
    features = []
    for i in range(len(states)):
        r, f = env_obj.reward_func(states[i][0], states[i][1], states[i][2], weights)
        rewards.append(r)
        features.append(f)
    # print "rewards is ", rewards
    value, policy = env_obj.value_iteration(rewards)
    # policy = np.random.randint(27, size=1331)
    print "policy is ", policy
    print "features is ", features
    # feat = np.array([features]).transpose().reshape((len(features[0]), len(features)))
    # print "features shape is ", feat.shape
    '''
    '''
    robot_mdp = RobotMarkovModel()
    # Finds the sum of features of the expert trajectory and list of all the features of the expert trajectory
    sum_trajectory_features, feature_array_all_trajectories = robot_mdp.generate_trajectories()
    svf = env_obj.compute_state_visitation_frequency(P_a, feature_array_all_trajectories, policy)
    print "svf is ", svf
    print "svf shape is ", svf.shape

    print "expected svf is ", feat.dot(svf).reshape(3, 1)


    # x = [-0.5, 0.2, 0.4]
    # row_column = obj_state_util.get_state_val_index(x)
    # print "index val", row_column, x
    # state_check = row_column
    # action_val = 15
    # print "Current state index ", obj_state_util.states[state_check]
    # r = obj_state_util.step(state_check, action_val)
    # print "r is ", r
    policy, state_traj, expected_svf = q_learning(env_obj, weights, alpha=0.1, gamma=0.9, epsilon=0.2)
    print "best policy is ", policy
    policy_val = []
    for i in range(len(policy)):
        policy_val.append(policy[i]/float(sum(policy)))
    print "policy val is ", policy_val
    # print "state traj", state_traj
    # print "rewards ", rewards
    # P_a = env_obj.get_transition_mat_deterministic()
    # print "prob is ", P_a
    # print "prob shape is ", P_a.shape
    # print "prob value is ", P_a[0]
    print "Expected svf is ", expected_svf
    '''



















