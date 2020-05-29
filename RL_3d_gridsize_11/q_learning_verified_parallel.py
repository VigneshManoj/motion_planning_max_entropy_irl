import numpy as np
from numpy import savetxt
from matplotlib import pyplot as plt
import concurrent.futures
from robot_markov_model import RobotMarkovModel


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
        # Initialize number of states and actions in the state space model created
        self.n_states = grid_size ** 3
        self.n_actions = 27
        self.gamma = discount

        self.rewards = np.zeros([self.n_states])
        # self.values_tmp = np.zeros([self.n_states])

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
            # reward = 1 if rewards[int(state)] > 1 else 0
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
        # reward = self.reward_func(resulting_state[0], resulting_state[1], resulting_state[2])
        # print "reward is ", reward
        # Checks if the resulting state is moving it out of the grid
        resulting_state_index = self.get_state_val_index(resulting_state)
        if not self.off_grid_move(resulting_state, self.states[curr_state]):
            reward = self.rewards[int(resulting_state_index)]
            return resulting_state, reward, self.is_terminal_state(resulting_state_index), None
        else:
            # If movement is out of the grid then just return the current state value itself
            # print "*****The value is moving out of the grid in step function*******"
            return self.states[curr_state], -1, self.is_terminal_state(resulting_state_index), None

    def action_space_sample(self):
        # print "random action choice ", np.random.randint(0, len(self.action_space))
        return np.random.randint(0, len(self.action_space))

global_state_val = 0
global_Q = {}

def calc_value_for_state(a):
    global global_Q, global_state_val
    # print "out func", global_state_val
    value = global_Q[global_state_val, a]
    # print "value of state is ", value
    return value, a

def max_action(Q, state_val, action_values):
    # print "max action action val ", action_values
    global global_Q, global_state_val
    actions_range_value = action_values
    # q_values = np.array([Q[state_val, a] for a in action_values])
    # print "values in max action is ", q_values
    global_Q = dict.copy(Q)
    # print "glbal q", global_Q[0, 0]
    global_state_val = state_val
    # print "inside func", global_state_val
    q_values = np.zeros(27)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        for q, a in executor.map(calc_value_for_state, actions_range_value):
            np.append(q_values, q)
    action = np.argmax([q_values])
    # print "---max action function action ", action
    # print "max a value ", action
    return action_values[action]


# def q_learning(env_obj, alpha, gamma, epsilon):
def q_learning(env_obj, alpha, gamma, epsilon):

    # env_obj = RobotStateUtils(11, weights)
    # states = env_obj.create_state_space_model_func()
    # action = env_obj.create_action_set_func()
    # print "State space created is ", states
    Q = {}
    num_games = 5000
    # highest_rew = 0
    total_rewards = np.zeros(num_games)
    # best_policy = []
    # Default value
    # most_reward_index = 0
    # sum_state_trajectory = 0
    # expected_svf = np.zeros(len(env_obj.states))
    # print "obj state ", env_obj.states.keys()
    # print "obj action ", env_obj.action_space.keys()
    for state in env_obj.states.keys():
        for action in env_obj.action_space.keys():
            Q[state, action] = 0

    for i in range(num_games):
        if i % 1 == 0:
            print('-------------starting game-------------- ', i)
        done = False
        ep_rewards = 0
        # episode_policy = []
        # state_trajectory = []
        observation = env_obj.reset()
        # visited_states = []

        # observation = 0
        count = 0

        while not done:
            rand = np.random.random()
            # print "random val is ", rand
            # print "----------------------------------------------------------------------------"
            # print "Starting state val inside loop ", observation
            # print "action val inside loop", env_obj.action_space.keys()
            action = max_action(Q, observation, env_obj.action_space.keys()) if rand < (1 - epsilon) \
                else env_obj.action_space_sample()
            observation_, reward, done, info = env_obj.step(observation, action)
            ep_rewards += reward
            next_observation_index = env_obj.get_state_val_index(observation_)
            # visited_states.append(next_observation_index)
            # print "Next obs index", next_observation_index
            action_ = max_action(Q, next_observation_index, env_obj.action_space.keys())
             #print "current action val is ", action
            # print "next action val is ", action_
            # print "reward is ", reward
            Q[observation, action] = Q[observation, action] + \
                                     alpha * (reward + gamma * Q[next_observation_index, action_] -
                                              Q[observation, action])
            # print "Q value in loop", Q[observation, action]
            # episode_policy.append((Q[observation, action]))
            # misc_val = Q[observation, action]
            # print "misc val1 ", Q[observation, action]
            # print "misc val2 ", alpha * (reward + gamma * Q[next_observation_index, action_] -
                                                                       # Q[observation, action])
            observation = next_observation_index
            # if count%100==0:
            #     print "state value after assigning to new state", env_obj.states[observation]
            # count += 1
            # state_trajectory.append(env_obj.states[observation])
        if epsilon - 2 / num_games > 0:
            epsilon -= 2 / num_games
        else:
            epsilon = 0
        total_rewards[i] = ep_rewards
        # if ep_rewards > highest_rew:
        #    highest_rew = ep_rewards
           # best_policy = episode_policy
        # policy[i] = episode_policy
        # most_reward_index = np.argmax(total_rewards)
        # policy_dict = {i: episode_policy}
        # policy.update(policy_dict)
        # state_dict = {i: state_trajectory}
        # state_trajectories.update(state_dict)
        # sum_state_trajectory = env_obj.sum_of_features(state_trajectories[most_reward_index])
    # expected_svf = env_obj.compute_state_visitation_freq(state_trajectories, policy[most_reward_index])

    return Q, total_rewards
    # return policy[most_reward_index], sum_state_trajectory, expected_svf


if __name__ == '__main__':
    # Robot Object called
    goal = np.array([0.005, 0.055, -0.125])
    # term_state = np.random.randint(0, grid_size ** 3)]
    # Pass the required gridsize, discount, terminal_state_val_from_trajectory):
    env_obj = RobotStateUtils(11, 0.01, goal)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    print "State space created is ", states
    # print np.argmax(rewards), np.max(rewards),
    policy = np.zeros(len(states))
    rewards = np.zeros(len(states))
    index = env_obj.get_state_val_index(goal)
    mdp_obj = RobotMarkovModel()
    trajectories = mdp_obj.generate_trajectories()
    index_vals = np.zeros(len(trajectories[0]))
    for i in range(len(trajectories[0])):
        # print "traj is ", trajectories[0][i]
        index_vals[i] = env_obj.get_state_val_index(trajectories[0][i])
    for _, ele in enumerate(index_vals):
        if ele == index:
            rewards[int(ele)] = 10
        else:
            rewards[int(ele)] = 1
    env_obj.rewards = rewards
    print "states is ", states[index-1], states[index], states[index+1]
    print index
    # print "actions are ", action

    Q, total_rew = q_learning(env_obj, alpha=0.1, gamma=0.01, epsilon=1)
    # print "Q is ", Q
    # print "Q shape is ", len(Q)
    # print "Q values are ", Q.values()
    # az = [Q[0, int(a)] for a in action]
    # plt.plot(total_rew)
    # plt.show()
    # plt.savefig("/home/vignesh/Desktop/reward_graph.png")
    # print "az is ", az
    for s in states:
        Q_for_state = [Q[int(s), int(a)] for a in action]
        # print "Q for each state is ", Q_for_state
        # print "state  ", s
        # policy[int(s)] = np.max(Q[int(s), int(a)] for a in action)
        policy[int(s)] = np.argmax(Q_for_state)
    print " policy is ", policy
    filename = "/home/vignesh/Desktop/individual_trials/version4/data1/policy_qlearning.txt"
    file_open = open(filename, 'a')
    savetxt(file_open, policy, delimiter=',', fmt="%10.5f", newline=", ")
    file_open.write("\n \n \n \n")
    file_open.close()
    # filename = "/home/vignesh/Desktop/individual_trials/version2/data3/best_policy_qlearning.txt"
    # file_op = open(filename, 'a')
    # savetxt(file_op, best_policy, delimiter=',', fmt="%10.5f", newline=", ")
    # file_op.write("\n \n \n \n")
    # file_op.close()
    # policy.tofile("/home/vignesh/Desktop/individual_trials/version2/data2/policy_qlearning.txt", sep=" ")





