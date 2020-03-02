import numpy as np
from numpy import savetxt
from matplotlib import pyplot as plt
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
            reward = 5
            # If terminal state is being given as a list then if state == self.terminal_state_val:
            # print "You have reached the terminal state "
            return reward, True
        else:
            reward = 1 if rewards[int(state)] > 1 else 0
            # reward = 1 if rewards[int(state)] > 1 else 0
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
            reward, done_val = self.is_terminal_state(resulting_state_index)
            return resulting_state, reward, done_val, None
        else:
            # If movement is out of the grid then just return the current state value itself
            # print "*****The value is moving out of the grid in step function*******"
            return self.states[curr_state], -1, False, None

    def action_space_sample(self):
        # print "random action choice ", np.random.randint(0, len(self.action_space))
        return np.random.randint(0, len(self.action_space))


def max_action(Q, state_val, action_values):
    # print "max action action val ", action_values
    q_values = np.array([Q[state_val, a] for a in action_values])
    # print "values in max action is ", q_values
    action = np.argmax(q_values)
    # print "---max action function action ", action
    # print "max q value ", q_values[action]
    return action_values[action]


# def q_learning(env_obj, alpha, gamma, epsilon):
def q_learning(env_obj, alpha, gamma, epsilon):

    # env_obj = RobotStateUtils(11, weights)
    # states = env_obj.create_state_space_model_func()
    # action = env_obj.create_action_set_func()
    # print "State space created is ", states
    Q = {}
    num_games = 5000
    highest_rew = 0
    total_rewards = np.zeros(num_games)
    # best_policy = []
    # Default value
    most_reward_index = 0
    sum_state_trajectory = 0
    # expected_svf = np.zeros(len(env_obj.states))
    # print "obj state ", env_obj.states.keys()
    # print "obj action ", env_obj.action_space.keys()
    for state in env_obj.states.keys():
        for action in env_obj.action_space.keys():
            Q[state, action] = 0

    for i in range(num_games):
        if i % 500 == 0:
            print('-------------starting game-------------- ', i)
        done = False
        ep_rewards = 0
        episode_policy = []
        state_trajectory = []
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
            # print "state value after assigning to new state", observation
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

rewards = np.array([0.99057,    0.16179,    0.72885,    0.93710,    0.41131,    0.22368,    0.35867,    0.96953,    0.24727,    0.62372,    0.39464,    0.13300,    0.17699,    0.09376,    0.35944,    0.57315,    0.84080,    0.46254,    0.72110,    0.36559,    0.34334,    0.82835,    0.64906,    0.48287,    0.90771,    0.72800,    0.44798,    0.13721,    0.01006,    0.80739,    0.22949,    0.36951,    0.14241,    0.05281,    0.35384,    0.29825,    0.69290,    0.71688,    0.14602,    0.92950,    0.94216,    0.62195,    0.69176,    0.13213,    0.99674,    0.34003,    0.64800,    0.10754,    0.64434,    0.77002,    0.35671,    0.61078,    0.92935,    0.98707,    0.56935,    0.91113,    0.51485,    0.55534,    0.74316,    0.52192,    0.39442,    0.07220,    0.45492,    0.90572,    0.40096,    0.97869,    0.52012,    0.82512,    0.44428,    0.65486,    0.94062,    0.30464,    0.99661,    0.76547,    0.63423,    0.39324,    0.43346,    0.65916,    0.40445,    0.96128,    0.26124,    0.98035,    0.78466,    0.63577,    0.20918,    0.89996,    0.12626,    0.67689,    0.01483,    0.21383,    0.70000,    0.99607,    0.16847,    0.67014,    0.88126,    0.54769,    0.48877,    0.63106,    0.50815,    0.01813,    0.82860,    0.11631,    0.83393,    0.81216,    0.36916,    0.81801,    0.26524,    0.89740,    0.77798,    0.58897,    0.87141,    0.43661,    0.82094,    0.85941,    0.60970,    0.82709,    0.77491,    0.28508,    0.66004,    0.76281,    0.48012,    0.24386,    0.54185,    0.89003,    0.18838,    0.25476,    0.15940,    0.23622,    0.25459,    0.73467,    0.68704,    0.36926,    0.11618,    0.25473,    4.83272,    5.29777,    0.74587,    0.19982,    0.37174,    0.32050,    0.34519,    0.90535,    0.27160,    0.34496,    0.70405,    0.05755,    0.55105,    0.02983,    0.53012,    0.59432,    0.25076,    0.12506,    0.98606,    0.02383,    0.78889,    0.45668,    0.98450,    0.22018,    0.87445,    0.83475,    0.41388,    0.24703,    0.91931,    0.27739,    0.92863,    0.54189,    0.19943,    0.20395,    0.25975,    0.90932,    0.35282,    0.12358,    0.75533,    0.45873,    0.64815,    0.55612,    0.86291,    0.00786,    0.05260,    0.99854,    0.06559,    0.21996,    0.36499,    0.87202,    0.43565,    0.52199,    0.01001,    0.18049,    0.01588,    0.13231,    0.82724,    0.56548,    0.36665,    0.83681,    0.19392,    0.35230,    0.92831,    0.76241,    0.13084,    0.01589,    0.15103,    0.48876,    0.04287,    0.78328,    0.43730,    0.90841,    0.52724,    0.64594,    0.36580,    0.09500,    0.33251,    0.65236,    0.38588,    0.38865,    0.95191,    0.16597,    0.92434,    0.57666,    0.12338,    0.32570,    0.01470,    0.92674,    0.29931,    0.59058,    0.48886,    0.61858,    0.01117,    0.91164,    0.70152,    0.33202,    0.92085,    0.59103,    0.97189,    0.76918,    0.71768,    0.48149,    0.26322,    0.33861,    0.59237,    0.07932,    0.95512,    0.59010,    0.65567,    0.20015,    0.66247,    0.73095,    0.69835,    0.01793,    0.67625,    0.16774,    0.72771,    0.45490,    0.05282,    0.96084,    0.16226,    5.62225,    0.44169,    0.55994,    0.09493,    0.54114,    0.82726,    0.70162,    0.42866,    0.34382,    0.73325,    0.85948,    5.38379,    5.79096,    0.27944,    0.83409,    0.82299,    0.19543,    0.26500,    0.96448,    0.78770,    0.04078,    0.40970,    0.89041,    0.96820,    0.83322,    0.37282,    0.18839,    0.66357,    0.53114,    0.41180,    0.96347,    0.82489,    0.57527,    0.30347,    0.76067,    0.77613,    0.37785,    0.84701,    0.65349,    0.31006,    0.04925,    0.32678,    0.68800,    0.49803,    0.07132,    0.12893,    0.40535,    0.82745,    0.14965,    0.69703,    0.95088,    0.13245,    0.43923,    0.99265,    0.98446,    0.35813,    0.24837,    0.92225,    0.23827,    0.71169,    0.05334,    0.22384,    0.24528,    0.39471,    0.67340,    0.61060,    0.09245,    0.87563,    0.96887,    0.66890,    0.18247,    0.27509,    0.00993,    0.10373,    0.53759,    0.47446,    0.55722,    0.79312,    0.20405,    0.13885,    0.29675,    0.85109,    0.97113,    0.14164,    0.40390,    0.28432,    0.64032,    0.55581,    0.49740,    0.79057,    0.36612,    0.16237,    0.85454,    0.72420,    0.08328,    0.79188,    0.29429,    0.79403,    0.95419,    0.13744,    0.19266,    0.53911,    0.31568,    0.52404,    0.51779,    0.28764,    0.41605,    0.95468,    0.72174,    0.20299,    5.10561,    5.70553,    0.29213,    0.46762,    0.94961,    0.77193,    0.62024,    0.05217,    0.93561,    0.96274,    0.51275,    5.38046,    0.43448,    0.53204,    0.11633,    0.18618,    0.68561,    0.07410,    0.79885,    0.27739,    0.82170,    0.34720,    5.78239,    0.47006,    0.86730,    0.77231,    0.64996,    0.99934,    0.49941,    0.78639,    0.23269,    0.50597,    0.27982,    5.28787,    0.86191,    0.18262,    0.93087,    0.20838,    0.79144,    0.38475,    0.94569,    0.37393,    0.53620,    0.60778,    0.62057,    0.40924,    0.89461,    0.17313,    0.82762,    0.70490,    0.54989,    0.98497,    0.58290,    0.11685,    0.80015,    0.70954,    0.89686,    0.13791,    0.08223,   -1.89503,    0.49281,    0.02869,    0.06855,    0.92963,    0.43488,    0.90043,    0.03768,    0.01252,    0.82742,    0.47199,    0.60917,    0.66319,    0.78479,    0.03509,    0.58399,    0.96313,    0.39296,    0.17812,    0.22287,    0.10496,    0.87843,    0.74372,    0.49694,    0.25288,    0.78079,    0.07045,    0.29527,    0.89067,    0.33540,    0.00747,    0.68207,    0.93866,    0.88860,    0.78624,    0.73380,    0.40862,    0.76666,    0.78193,    0.00835,    0.28769,    0.36853,    0.93232,    0.36765,    0.07310,    0.02724,    0.29315,    0.51982,    0.98470,    0.27219,    0.72264,    0.48481,    0.59773,    0.71694,    0.39421,    0.02970,    0.92924,    0.86378,    0.76223,    0.09889,    0.49454,    0.25705,    0.79082,   10.40506,    5.45538,    0.14541,    0.32459,    0.73247,    0.22466,    0.93729,    0.34659,    0.71208,    0.31904,    0.57545,    4.93066,    5.11019,    0.32788,    0.37344,    0.66595,    0.74145,    0.41264,    0.10369,    0.04506,    0.09167,    0.84960,    0.94403,    0.74949,    0.48755,    0.40756,    0.99517,    0.21890,    0.37571,    0.53656,    0.93618,    4.88006,    5.54858,    0.12365,    0.65912,    0.47946,    0.98962,    0.41331,    0.51544,    0.48983,    0.35665,    5.55826,   10.03414,    0.26313,    0.63790,    0.72387,    0.59570,    0.60711,    0.39692,    0.24448,    0.82162,    0.43651,    0.05457,    0.19416,    0.11962,    0.83980,    0.80350,    0.10200,   -2.05124,    0.06710,    0.20295,    0.57754,    0.69244,    0.80825,    0.80069,    0.39456,    0.93974,    0.34343,    0.40930,    0.75435,    0.70692,    0.77810,    0.28694,    0.14000,    0.42209,    0.17087,    0.35480,    0.69622,    0.51505,    0.17517,    0.60193,    0.10171,    0.20671,    0.43246,    0.98313,    0.40924,    0.48686,    0.65876,    0.67786,    0.13041,    0.99489,    0.90409,    0.40879,    0.93001,    0.77334,    0.78640,    0.47450,    0.29861,    0.14616,    0.41210,    0.91929,    0.49372,    0.39488,    0.42769,    0.91334,    0.58161,    0.89637,    0.71462,    0.67239,    0.05629,    0.92405,    0.25767,    0.44889,    0.56841,    0.39221,    0.12402,    0.96690,    0.26716,    0.90181,    0.21972,    0.42627,    5.04672,    0.64539,    0.07741,    0.70336,    0.42858,    0.23501,    0.97724,    0.95153,    0.52305,    0.91933,    0.00413,    5.21428,    0.28103,    0.34800,    0.40497,    0.09920,    0.76345,    0.25837,    0.28935,    0.65811,    0.19594,    0.13705,    0.32012,    0.23581,    0.79756,    0.49132,    0.40495,    0.11719,    0.81704,    0.39528,    0.61751,    0.42753,    0.77606,    0.66989,    0.03130,    0.19077,    0.14382,    0.96161,    0.95502,    0.80828,    0.80119,    0.86595,    0.42939,    0.28881,    0.83581,    0.63056,    0.59100,    0.23461,    0.57732,    0.08011,    0.09311,    0.96248,    0.85544,    0.90115,    0.42014,    0.61472,    0.65141,    0.00878,   -1.50971,    0.28473,    0.40875,    0.28330,    0.16575,    0.95783,    0.17461,    0.08136,    0.24391,    0.85914,    0.85269,    0.82855,    0.42977,    0.23463,    0.70826,    0.94817,    0.22895,    0.07763,    0.33327,    0.15529,    0.25420,    0.83716,    0.65032,    0.00390,    0.27464,    0.79216,    0.84118,    0.36666,    0.01035,    0.11520,    0.22671,    0.05026,    0.10434,    0.58424,    0.71892,    0.71705,    0.81529,    0.94620,    0.62586,    0.46194,    0.27932,    0.55311,    0.68539,    0.78178,    0.35813,    0.25208,    0.98256,    0.29758,    0.85068,    0.95393,    0.37503,    0.93651,    0.45261,    0.84706,    0.13795,    0.65718,    0.05701,    0.95137,    0.89385,    0.24515,    0.59320,    0.20726,    0.37148,    0.75008,    0.63155,    0.29556,    0.16381,    0.04797,    0.14815,    0.33275,    0.07039,    0.75801,    0.42221,    0.40578,    5.13980,    0.08068,    0.24832,    0.57598,    0.75847,    0.42543,    0.42653,    0.58844,    0.44735,    0.53185,    5.10622,    5.00358,    0.55985,    0.77451,    0.93035,    0.38991,    0.68991,    0.59819,    0.46756,    0.00492,    0.85361,    5.36152,    0.72959,    0.45866,    0.39324,    0.17573,    0.04774,    0.16726,    0.68309,    0.64011,    0.53050,    0.13928,    0.16679,    0.75551,    0.82039,    0.31188,    0.01196,    0.18378,    0.68974,    0.21572,    0.22357,    0.57964,    0.15613,    0.32778,    0.61410,    0.27372,   -1.92653,    0.21111,    0.84540,    0.40083,    0.81204,    0.17245,    0.83307,    0.29393,    0.57619,    0.70979,    0.28383,    0.70740,    0.28377,    0.24710,    0.04845,    0.59740,    0.90048,    0.85393,    0.38273,    0.11351,    0.25537,    0.75860,    0.27968,    0.86262,    0.97174,    0.74489,    0.41769,    0.83958,    0.26680,    0.36204,    0.78677,    0.12872,    0.75757,    0.91998,    0.55226,    0.70417,    0.59720,    0.37717,    0.84568,    0.26587,    0.62852,    0.78223,    0.94170,    0.07440,    0.67059,    0.01576,    0.54450,    0.22737,    0.66931,    0.60035,    0.30944,    0.18352,    0.01457,    0.38277,    0.67257,    0.03636,    0.89919,    0.35094,    0.70210,    0.76471,    0.35403,    0.78884,    0.53890,    0.35885,    0.87336,    0.67017,    0.66015,    0.95743,    0.05500,    0.52236,    0.10944,    0.87437,    0.05537,    0.81056,    0.00429,    0.65244,    0.28767,    0.42208,    0.25380,    0.30193,    0.06489,    0.00465,    0.22321,    0.28220,    0.66905,    0.98186,    0.21855,    0.69597,    0.68204,    0.95573,    0.29909,    0.79916,    0.75417,    0.19593,    0.52253,    0.89956,    4.93280,    0.36453,    0.04645,    0.53266,    0.92017,    0.23002,    0.41619,    0.43672,    0.07107,    0.72693,    5.42631,    9.94825,    0.42812,    0.14011,    0.19106,    0.20213,    0.83932,    0.38417,    0.26772,    0.06275,    0.17967,    0.06208,    0.27440,    0.08845,    0.37706,    0.52749,    0.93581,    0.48566,    0.95061,    0.66812,    0.03849,    0.76985,    0.31841,    0.27435,    0.94597,    0.63725,    0.40202,    0.31119,    0.83247,    0.86984,    0.79922,    0.88012,    0.64060,    0.23677,    0.54828,    0.20128,    0.21046,    0.25391,    0.04529,    0.76613,    0.17785,    0.43336,    0.51945,    0.73146,    0.77982,    0.50602,    0.99979,    0.05009,    0.59857,    0.22244,    0.85635,    0.30312,    0.70469,    0.40058,    0.99529,    0.15502,    0.25233,    0.97544,    0.88246,    0.43775,    0.35848,    0.56517,    0.73764,    0.84995,    0.87835,    0.04805,    0.69517,    0.99431,    0.03798,    0.48013,    0.63459,    0.68341,    0.65511,    0.17035,    0.65470,    0.32850,    0.54428,    0.36674,    0.71429,    0.55929,    0.20986,    0.04273,    0.60041,    0.67462,    0.82826,    0.43809,    0.13537,    0.18350,    0.40798,    0.02431,    0.28568,    0.59309,    0.15540,    0.96110,    0.48361,    0.56548,    0.87582,    0.85630,    0.95571,    0.39141,    0.26689,    0.50302,    0.26617,    0.47735,    0.22879,    0.36098,    0.29049,    0.19788,    0.93035,    0.19018,    0.55313,    0.65623,    0.58253,    0.41962,    0.55150,    0.76710,    0.86459,    0.92935,    0.13461,    0.61295,    0.64828,    0.97587,    0.29602,    0.86304,    0.75268,    0.34442,    0.40337,    0.54522,    0.35678,    0.15567,    0.11874,    0.88384,    0.88494,    0.48133,    0.74080,    0.11611,    0.60086,    0.14229,    0.36163,    0.36196,    0.77387,    0.21355,    0.76387,    0.11501,    0.16729,    0.50397,    0.05950,    0.76670,    0.70038,    0.33644,    0.04456,    0.76390,    0.21408,    0.05131,    0.30882,    0.92801,    0.16085,    0.19912,    0.63642,    0.38799,    0.03592,    0.05150,    0.59390,    0.00872,    0.73228,    0.74984,    0.70388,    0.06496,    0.53192,    0.65596,    0.58502,    0.52526,    0.79753,    0.79197,    0.78461,    0.96658,    0.35246,    0.94728,    0.28227,    0.53293,    0.77317,    0.63221,    0.20969,    0.40631,    0.47352,    0.95808,    0.32209,    0.17553,    0.77497,    0.73877,    0.63823,    0.60045,    0.69401,    0.64890,    0.20996,    0.81767,    0.37660,    0.55058,    0.64339,    0.33195,    0.71367,    0.98069,    0.65216,    0.34040,    0.22493,    0.47256,    0.65824,    0.55773,    0.80574,    0.99625,    0.53954,    0.71044,    0.14945,    0.34579,    0.74363,    0.62627,    0.71146,    0.75473,    0.90289,    0.67819,    0.75724,    0.10633,    0.87992,    0.16142,    0.47879,    0.62376,    0.75082,    0.08527,    0.65571,    0.50037,    0.58971,    0.55223,    0.46187,    0.37754,    0.08321,    0.89141,    0.93069,    0.17029,    0.44771,    0.28522,    0.59531,    0.19280,    0.68227,    0.84126,    0.94962,    0.84649,    0.65753,    0.78739,    0.82397,    0.14554,    0.55561,    0.07868,    0.89239,    0.37528,    0.50200,    0.96923,    0.51411,    0.16853,    0.07151,    0.78812,    0.61531,    0.00161,    0.14647,    0.60405,    0.90597,    0.15656,    0.42308,    0.04964,    0.98044,    0.21249,    0.46026,    0.86381,    0.41540,    0.48864,    0.43683,    0.43918,    0.32761,    0.45513,    0.61832,    0.55164,    0.60170,    0.70785,    0.16383,    0.16747,    0.60609,    0.80761,    0.86620,    0.50124,    0.67104,    0.88759,    0.48666,    0.57342,    0.41452,    0.59852,    0.51846,    0.44577,    0.10415,    0.97442,    0.56257,    0.35604,    0.30132,    0.78355,    0.19644,    0.35505,    0.62089,    0.20419,    0.61078,    0.21685,    0.50566,    0.20302,    0.43365,    0.75291,    0.16155,    0.49749,    0.27099,    0.85535,    0.05507,    0.35527,    0.24303,    0.90415,    0.20988,    0.02881,    0.40227,    0.78479,    0.07644,    0.77770,    0.10185,    0.42403,    0.73830,    0.29491,    0.57447,    0.50307,    0.02004,    0.78448,    0.82895,    0.22270,    0.61050,    0.70842,    0.62614,    0.68926,    0.37096,    0.33725,    0.78148,    0.01648,    0.16052,    0.27737,    0.32912,    0.08379,    0.60278,    0.65785,    0.71668,    0.83060,    0.58629,    0.38034,    0.99750,    0.23417,    0.45598,    0.05330,    0.23558,    0.16673,    0.57564,    0.40983,    0.10031,    0.71449,    0.07212,    0.01169,    0.48008,    0.57174,    0.87879,    0.49703,    0.52723,    0.27888,    0.14338,    0.17276,    0.45165,    0.26168,    0.06020,    0.58559,    0.41699,    0.07377,    0.21029,    0.89163,    0.13092,    0.28196,    0.20386,    0.56454,    0.12290,    0.44554,    0.51450,    0.43226,    0.83742,    0.65245,    0.85158,    0.10584,    0.63598,    0.69971,    0.90389,    0.07057,    0.73070,    0.30497,    0.98563,    0.91180,    0.31903,    0.76609,    0.11650,    0.40296,    0.06630,    0.26481,    0.01434,    0.84020,    0.22687,    0.48736,    0.53480,    0.11452,    0.33375,    0.18248,    0.09581,    0.85239,    0.00983,    0.46776,    0.21988,    0.35504,    0.53449,    0.71291,    0.43543,    0.21619,    0.26979,    0.02408,    0.29454,    0.68883,    0.09901,    0.38798,    0.72609,    0.35661,    0.08401,    0.24404,    0.88496,    0.81184,    0.44084])

if __name__ == '__main__':
    # Robot Object called
    goal = np.array([0.01, 0.05, -0.13])
    # term_state = np.random.randint(0, grid_size ** 3)]
    # Pass the required gridsize, discount, terminal_state_val_from_trajectory):
    env_obj = RobotStateUtils(11, 0.9, goal)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    print "State space created is ", states[487]
    print np.argmax(rewards)
    policy = np.zeros(len(states))
    index = env_obj.get_state_val_index(goal)
    print "states is ", states[index-1], states[index], states[index+1]
    print index
    # print "actions are ", action

    Q, total_rew = q_learning(env_obj, alpha=0.1, gamma=0.9, epsilon=1)
    # print "Q is ", Q
    # print "Q shape is ", len(Q)
    # print "Q values are ", Q.values()
    # az = [Q[0, int(a)] for a in action]
    plt.plot(total_rew)
    plt.show()
    plt.savefig("/home/vignesh/Desktop/reward_graph.png")
    # print "az is ", az
    for s in states:
        Q_for_state = [Q[int(s), int(a)] for a in action]
        # print "Q for each state is ", Q_for_state
        # print "state  ", s
        # policy[int(s)] = np.max(Q[int(s), int(a)] for a in action)
        policy[int(s)] = np.argmax(Q_for_state)
    print " policy is ", policy
    filename = "/home/vignesh/Desktop/individual_trials/version2/data2/policy_qlearning.txt"
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





