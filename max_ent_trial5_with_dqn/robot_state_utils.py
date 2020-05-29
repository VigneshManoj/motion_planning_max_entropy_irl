import numpy as np
import numba as nb
import math
import concurrent.futures
from robot_markov_model import RobotMarkovModel
import numpy.random as rn
from pytictoc import TicToc


class RobotStateUtils(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, grid_size, discount, terminal_state_val_from_trajectory):
        super(RobotStateUtils, self).__init__(max_workers=8)
        # Model here means the 3D cube being created
        # linspace limit values: limit_values_pos = [[-0.009, -0.003], [0.003, 007], [-0.014, -0.008]]
        # Creates the model state space based on the maximum and minimum values of the dataset provided by the user
        # It is for created a 3D cube with 3 values specifying each cube node
        # The value 11 etc decides how sparse the mesh size of the cube would be
        self.grid_size = grid_size
        self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        self.lin_space_limits = np.linspace(-0.05, 0.05, self.grid_size, dtype='float32')
        # Creates a dictionary for storing the state values
        self.states = {}
        # Creates a dictionary for storing the action values
        self.action_space = {}
        # Numerical values assigned to each action in the dictionary
        self.possible_actions = [i for i in range(27)]
        # Total Number of states defining the state of the robot
        self.n_params_for_state = 3
        # The terminal state value which is taken from the expert trajectory data
        self.terminal_state_val = terminal_state_val_from_trajectory
        # Deterministic or stochastic transition environment
        self.trans_prob = 1
        # Initialize number of states and actions in the state space model created
        self.n_states = grid_size**3
        # The number of actions present in the environment
        self.n_actions = 27
        # The discount factor being used
        self.gamma = discount
        # Initialize the rewards array
        self.rewards = []
        # Initialize the transition matrix with shape (Number of states x number of actions x number of states)
        self.transition_matrix = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.int32)
        # Initialize the copy of values for value iteration
        # self.values_tmp = np.zeros([self.n_states])



    def create_state_space_model_func(self):
        # Creates the state space of the robot based on the values initialized for linspace by the user
        print "lin space is ", self.lin_space_limits
        # print "Creating State space "
        state_set = []
        for i_val in self.lin_space_limits:
            for j_val in self.lin_space_limits:
                for k_val in self.lin_space_limits:
                    # Rounding state values so that the values of the model, dont take in too many floating points
                    state_set.append([round(i_val, 3), round(j_val, 3), round(k_val, 3)])
        # Assigning the dictionary keys
        for i in range(len(state_set)):
            state_dict = {i: state_set[i]}
            self.states.update(state_dict)

        return self.states

    def create_action_set_func(self):
        # Creates the action space required for the robot. It is defined by the user beforehand itself
        action_set = []
        for pos_x in [-0.01, 0, 0.01]:
            for pos_y in [-0.01, 0, 0.01]:
                for pos_z in [-0.01, 0, 0.01]:
                    action_set.append([pos_x, pos_y, pos_z])
        # Assigning the dictionary keys
        for i in range(len(action_set)):
            action_dict = {i: action_set[i]}
            self.action_space.update(action_dict)

        return self.action_space

    # Creates a feature matrix based on the distance from neighbouring points for creating the feature map
    def get_feature_matrix(self):
        n = self.grid_size**3
        feature = np.zeros([n, n])
        for i in range(n):
            for x in self.lin_space_limits:
                for y in self.lin_space_limits:
                    for z in self.lin_space_limits:
                        ix, iy, iz = self.states[i]
                        # print "index val is ", self.get_state_val_index([x, y, z]), "xyz is ", [x, y, z]
                        feature[i, int(self.get_state_val_index([x, y, z]))] = abs(ix - x) + abs(iy - y) + abs(iz - z)
                        # print "feature is ", feature[i, int(self.get_state_val_index([x, y, z]))]
        return feature

    # To convert a state value array into index values
    def get_state_val_index(self, state_val):
        # Since the starting is 0.05, we multiply it by 10 and then add 0.5 which makes it zero
        index_val = abs((state_val[0]*10 + 0.5) * pow(self.grid_size, 2)) + \
                    abs((state_val[1]*10 + 0.5) * pow(self.grid_size, 1)) + \
                    abs((state_val[2]*10 + 0.5))
        # Returns the index value
        return round(index_val*(self.grid_size-1))

    # Checks if the terminal state has been reached
    def is_terminal_state(self, state):

        # because terminal state is being given in array value and needs to convert to index value
        terminal_state_val_index = self.get_state_val_index(self.terminal_state_val)
        if state == terminal_state_val_index:
            # If terminal state is being given as a list then if state == self.terminal_state_val:
            # print "You have reached the terminal state "
            return True
        else:
            # It has not yet reached the terminal state
            return False

    # Checks if the action will result in a movement outside the created environment
    def off_grid_move(self, new_state, old_state):

        # Checks if the new state exists in the state space
        if new_state not in self.states.values():
            return True
        # if trying to wrap around the grid, also the reason for the for x in _ is because old_state is a list
        elif (x % self.grid_size for x in old_state) == 0 and (y % self.grid_size for y in new_state) == self.grid_size - 1:
            return True
        elif (x % self.grid_size for x in old_state) == self.grid_size - 1 and (y % self.grid_size for y in new_state) == 0:
            return True
        else:
            # If there are no issues with the new state value then return false, negation is present on the other end
            return False
    '''
    def exp_reward_func(self, end_pos_x, end_pos_y, end_pos_z, weights):
        # Creates list of all the features being considered
        features = [self.features_array_prim_func, self.features_array_sec_func, self.features_array_tert_func]
        reward = 0
        features_arr = []
        for n in range(0, len(features)):
            features_arr.append(features[n](end_pos_x, end_pos_y, end_pos_z))

            reward = reward + weights[0, n]*features_arr[n]
        # Created the feature function assuming everything has importance, so therefore added each parameter value
        # return reward, np.array([features_arr]), len(features)
        return reward, features_arr


    def reward_func(self, end_pos_x, end_pos_y, end_pos_z, alpha):
        # Creates list of all the features being considered

        # reward = -1
        if self.is_terminal_state([end_pos_x, end_pos_y, end_pos_z]):
            reward = 0
        else:
            reward = -1

        return reward, None


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
        self.grid = np.zeros((self.grid_size, self.grid_size, self.grid_size))
        return self.pos

    def step(self, curr_state, action):
        resulting_state = []
        # print "current state", self.states[curr_state]
        # print "action taken", action, self.action_space[action]
        # Finds the resulting state when the action is taken at curr_state
        for i in range(0, self.n_params_for_state):
            resulting_state.append(round(self.states[curr_state][i] + self.action_space[action][i], 1))

        # Checks if the resulting state is moving it out of the grid
        resulting_state_index = self.get_state_val_index(resulting_state)
        if not self.off_grid_move(resulting_state, self.states[curr_state]):
            return resulting_state, self.is_terminal_state(resulting_state_index), None
        else:
            # If movement is out of the grid then just return the current state value itself
            # print "*****The value is moving out of the grid in step function*******"
            return self.states[curr_state], self.is_terminal_state(curr_state), None

    def action_space_sample(self):
        # print "random action choice ", np.random.randint(0, len(self.action_space))
        return np.random.randint(0, len(self.action_space))
    '''
    def features_func(self, end_pos_x, end_pos_y, end_pos_z):

        features = [self.features_array_prim_func, self.features_array_sec_func, self.features_array_tert_func]
        features_arr = []
        for n in range(0, len(features)):
            features_arr.append(features[n](end_pos_x, end_pos_y, end_pos_z))
        # Created the feature function assuming everything has importance, so therefore added each parameter value
        return features_arr
    '''
    def get_transition_states_and_probs(self, curr_state, action):

        if self.is_terminal_state(curr_state):
            return [(curr_state, 1)]
        resulting_state = []
        if self.trans_prob == 1:
            for i in range(0, self.n_params_for_state):
                resulting_state.append(round(self.states[curr_state][i] + self.action_space[action][i], 1))
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
                    self.transition_matrix[si, a, sj] = prob
        return self.transition_matrix

    '''
    def calc_value_for_state(self, s):
        value = max([sum([self.transition_matrix[s, a, s1] * (self.rewards[s] + self.gamma * self.values_tmp[s1]) for s1 in range(self.n_states)])
                     for a in range(self.n_actions)])
        return value, s

    def value_iteration(self, rewards, error=0.1):
        # Initialize the value function
        t_complete_func = TicToc()
        t_complete_func.tic()
        values = np.zeros([self.n_states])
        states_range_value = range(0, self.n_states)
        print "states range value is ", states_range_value
        self.rewards = rewards
        # estimate values
        while True:
            # Temporary copy to check find the difference between new value function calculated & current value function
            # to ensure improvement in value
            self.values_tmp = values.copy()
            t_value = TicToc()
            t_value.tic()
            for q, s in self.map(self.calc_value_for_state, states_range_value):
                values[s] = q
                # print "values is ", values[s]
            
            # for s in range(self.n_states):
            #     values[s] = max(
            #         [sum([transition_matrix[s, a, s1] * (rewards[s] + self.gamma * values_tmp[s1])
            #               for s1 in range(self.n_states)])
            #          for a in range(self.n_actions)])
            
            t_value.toc('Value function section took')
                # print "values ", values[s]
            if max([abs(values[s] - self.values_tmp[s]) for s in range(self.n_states)]) < error:
                break
        # generate deterministic policy
        policy = np.zeros([self.n_states])
        for s in range(self.n_states):
            policy[s] = np.argmax([sum([self.transition_matrix[s, a, s1] * (self.rewards[s] + self.gamma * values[s1])
                                        for s1 in range(self.n_states)])
                                   for a in range(self.n_actions)])

        t_complete_func.toc('Complete function section took')
        return values, policy
    '''

    # Calculates the state visitation frequency using the transition matrix
    def compute_state_visitation_frequency(self, trajectories, optimal_policy):
        n_trajectories = len(trajectories)
        total_states = len(trajectories[0])
        # d_states = len(trajectories[0][0])
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
        # Find the average mu value
        mu[:, 0] = mu[:, 0] / n_trajectories

        # Based on the algorithm mentioned in the paper
        for s in range(self.n_states):
            for t in range(T - 1):
                # Computes the mu value for each state once the optimal action is taken
                mu[s, t + 1] = sum([mu[pre_s, t] * self.transition_matrix[pre_s, int(optimal_policy[pre_s]), s]
                                    for pre_s in range(self.n_states)])
        p = np.sum(mu, 1)
        # Returns the state visitation frequency
        return p


# Find the action value which has the maximum value for a given a state value function
def max_action(Q, state_val, action_values):
    # print "max action action val ", action_values
    q_values = np.array([Q[state_val, a] for a in action_values])
    # print "values in max action is ", q_values
    action = np.argmax(q_values)
    # print "---max action function action ", action
    # print "max q value ", q_values[action]
    return action_values[action]


# Q learning algorithm
def q_learning(env_obj, reward, alpha, gamma):

    Q = {}
    # Decides the number of episodes to run the q_learning algorithm
    num_games = 1500
    # Initialize the total reward being collected
    total_rewards = np.zeros(num_games)
    # Decides how many times the action selected will be randomly (exploration and exploitation)
    epsilon = 0.2
    # Initialize the Q value
    for state in env_obj.states.keys():
        for action in env_obj.action_space.keys():
            Q[state, action] = 0
    # Run the algorithm the number of times specified by the user
    for i in range(num_games):
        # Decides how often it would be printed to give updates
        if i % 250 == 0:
            print('-------------starting game-------------- ', i)
        # Starts with assuming terminal state hasn't been reached
        done = False
        # Initialize episode reward
        ep_rewards = 0
        # Resets the observations for each episode and assigns a a random starting value for each episode
        observation = env_obj.reset()

        # Until the terminal state is reached
        while not done:
            rand = np.random.random()
            # print "random val is ", rand
            # print "----------------------------------------------------------------------------"
            # print "Starting state val inside loop ", observation
            # print "action val inside loop", env_obj.action_space.keys()
            # The action is selected which has the maximum Q value
            action = max_action(Q, observation, env_obj.action_space.keys()) if rand < (1 - epsilon) \
                else env_obj.action_space_sample()
            # Computes the next state reached if the action found above is taken and checks if its terminal state
            observation_, done, info = env_obj.step(observation, action)
            # Finds the index value for the resulting observation state reached
            next_observation_index = env_obj.get_state_val_index(observation_)
            # Updates the episode rewards
            ep_rewards += reward[int(next_observation_index)]
            # Computes the next best action that can be taken from the resulting state (highest q value)
            action_ = max_action(Q, next_observation_index, env_obj.action_space.keys())
            # Q value is updated based on the discount factor and learning rate using the q learning algorithm
            Q[observation, action] = Q[observation, action] + \
                                     alpha * (reward[int(next_observation_index)] +
                                              gamma * Q[next_observation_index, action_] -
                                              Q[observation, action])
            # print "Q value in loop", Q[observation, action]
            # The resulting state is the starting state for the next step
            observation = next_observation_index
            # print "state value after assigning to new state", observation
            # state_trajectory.append(env_obj.states[observation])
        # Updates the epsilon value
        if epsilon - 2 / num_games > 0:
            epsilon -= 2 / num_games
        else:
            epsilon = 0
            # Stores the rewards collected by individual episodes
        total_rewards[i] = ep_rewards

    # Returns the q value for each state action for finding the optimal policy
    return Q

# Computes the optimal policy given the q values for each state action
def optimal_policy_func(states, action, env_obj, reward, learning_rate, discount):
    # Calls the q_learning function so that each state action values can be found
    Q = q_learning(env_obj, reward, learning_rate, discount)
    # Initialize the policy based on the number of states present in the environment (find best action from each state)
    policy = np.zeros(len(states))
    # For each state
    for s in states:
        # Find the q values for each actions which can be taken from a specific state
        Q_for_state = [Q[int(s), int(a)] for a in action]
        # print "Q for each state is ", Q_for_state
        # print "state  ", s
        # policy[int(s)] = np.max(Q[int(s), int(a)] for a in action)
        # Choose the action which returns the maximum q value
        policy[int(s)] = np.argmax(Q_for_state)
    # print " policy is ", policy
    # Returns the optimal policy
    return policy


if __name__ == '__main__':
    # Robot Object called
    # Pass the gridsize required
    weights = np.array([[1, 1, 0]])
    # term_state = np.random.randint(0, grid_size ** 3)]
    env_obj = RobotStateUtils(11, weights, 0.9)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    print "State space created is ", states
    print "State space created is ", len(states)
    print "actions is ", action

    '''
    transition_matrix = env_obj.get_transition_mat_deterministic()
    # print "transition_matrix is ", transition_matrix
    print "shape of transition_matrix ", transition_matrix.shape
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

    
    robot_mdp = RobotMarkovModel()
    # Finds the sum of features of the expert trajectory and list of all the features of the expert trajectory
    sum_trajectory_features, feature_array_all_trajectories = robot_mdp.generate_trajectories()
    svf = env_obj.compute_state_visitation_frequency(transition_matrix, feature_array_all_trajectories, policy)
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
    # transition_matrix = env_obj.get_transition_mat_deterministic()
    # print "prob is ", transition_matrix
    # print "prob shape is ", transition_matrix.shape
    # print "prob value is ", transition_matrix[0]
    print "Expected svf is ", expected_svf
    '''



















