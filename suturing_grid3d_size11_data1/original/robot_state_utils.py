import numpy as np
import math
import concurrent.futures
from robot_markov_model import RobotMarkovModel
import numpy.random as rn
from pytictoc import TicToc
from numpy import savetxt


class RobotStateUtils(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, grid_size, discount, terminal_state_val_from_trajectory):
        super(RobotStateUtils, self).__init__(max_workers=48)
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
        self.n_states = grid_size**3
        self.n_actions = 27
        self.gamma = discount
        self.temp_state = 0
        self.rewards = np.zeros([self.n_states])

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
        elif (x % self.grid_size for x in old_state) == 0 and (y % self.grid_size for y in new_state) == self.grid_size - 1:
            return True
        elif (x % self.grid_size for x in old_state) == self.grid_size - 1 and (y % self.grid_size for y in new_state) == 0:
            return True
        else:
            # If there are no issues with the new state value then return false, negation is present on the other end
            return False

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
            reward = self.rewards[int(resulting_state_index)]
            return resulting_state_index, reward
        else:
            # If movement is out of the grid then just return the current state value itself
            # print "*****The value is moving out of the grid in step function*******"
            return curr_state, -1

    def calc_value_for_state(self, a):
        resulting_state_index, reward = self.step(self.temp_state, a)
        value = reward + self.gamma * V[resulting_state_index]
        return value, a

    def iterateValues(self, V, policy, error):
        converged = False
        i = 0
        action_range_value = range(0, self.n_actions)

        while not converged:
            DELTA = 0
            for state in self.states.keys():
                # print "state is ", state
                i += 1
                oldV = V[int(state)]
                # print oldV
                newV = np.zeros([self.n_states])
                self.temp_state = state
                for q, a in self.map(self.calc_value_for_state, action_range_value):
                    newV[state] = q
                # newV = np.array(newV)

                bestV = np.where(newV == newV.max())[0]
                # print "max is ", newV.max()
                # if i%25 ==0:
                #    print "new state is ", newV
                #    print "best value is ", bestV
                bestState = np.random.choice(bestV)
                V[state] = newV[bestState]
                DELTA = max(DELTA, np.abs(oldV - V[state]))
                converged = True if DELTA < error else False

        for state in self.states.keys():
            newValues = []
            actions = []
            i += 1
            for action in self.action_space.keys():
                resulting_state_index, reward = self.step(state, action)
                newValues.append(reward + self.gamma * V[resulting_state_index])
                actions.append(action)
            newValues = np.array(newValues)
            bestActionIDX = np.where(newValues == newValues.max())[0]
            bestActions = actions[bestActionIDX[0]]
            policy[state] = bestActions
        print(i, 'sweeps of state space for value iteration')
        return V, policy


if __name__ == '__main__':
    goal = np.array([0.005, 0.055, -0.125])
    # term_state = np.random.randint(0, grid_size ** 3)]
    env_obj = RobotStateUtils(11, 0.9, goal)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    print "State space created is ", states
    print "actions is ", action
    index_val = env_obj.get_state_val_index(goal)
    print "index val is ", index_val
    mdp_obj = RobotMarkovModel()
    trajectories = mdp_obj.generate_trajectories()
    index_vals = np.zeros(len(trajectories[0]))
    # rewards = np.zeros(len(states))
    rewards = np.array([7.54223,    7.13118,    6.78729,    6.16313,    5.98842,    7.04816,    7.57567,    7.62666,    8.80041,    9.61791,    9.95611,    6.62645,    6.28106,    5.50881,    4.85437,    4.98816,    5.56522,    6.59340,    7.51678,    8.48532,    8.82148,    9.87299,    5.90261,    5.24482,    5.24086,    4.14796,    4.27847,    4.96861,    6.40135,    6.98284,    7.56000,    8.33879,    8.80249,    6.39573,    5.58079,    5.03519,    4.26906,    4.75917,    5.57157,    6.01481,    6.16807,    7.04196,    8.25130,    8.61474,    6.48133,    5.05824,    5.16921,    4.57523,    4.34306,    5.26094,    5.89430,    6.15558,    6.78633,    7.47348,    8.69935,    6.75831,    5.41029,    4.82176,    4.85539,    5.00008,    5.52066,    5.93534,    7.11725,    7.61816,    8.15622,    8.56487,    6.67740,    6.69390,    5.67311,    4.81960,    4.74061,    5.52939,    6.96912,    7.44398,    8.30728,    8.58763,    9.76565,    7.28875,    6.92646,    6.54106,    5.50925,    5.64711,    6.40621,    7.49507,    7.59486,    8.47409,    9.56801,   10.35415,    7.93486,    7.32626,    7.18189,    6.00335,    6.74533,    7.43622,    7.97101,    9.09052,    8.98743,    9.47702,   10.67435,    8.72125,    8.30984,    7.12618,    6.82285,    6.93953,    7.51979,    8.94352,    8.92826,    9.69475,   10.76377,   11.40683,    9.78239,    9.00176,    7.76331,    7.94762,    7.53444,    8.18909,    9.06648,   10.23423,   10.65371,   11.69839,   12.18922,    6.73165,    6.43806,    5.97389,    5.68331,    5.73079,    6.32882,    6.83633,    7.48463,    7.84718,    8.61898,    9.21620,    6.57456,    5.46284,    5.43449,    4.38526,    4.95110,    5.76689,    6.38504,    6.74323,    7.47551,    7.78911,    9.02087,    5.85052,    4.81708,    4.79315,    3.94789,    4.32121,    5.21960,    5.20951,    5.87734,    6.47558,    7.45647,    8.61909,    5.70868,    5.20993,    3.78590,    4.13122,    3.75512,    4.84013,    5.58815,    5.78059,    6.36148,    7.67912,    8.37552,    5.49436,    4.63241,    4.11481,    4.19832,    4.21798,    4.22087,    5.07394,    5.58687,    6.31570,    7.42601,    8.33055,    5.34942,    5.45483,    3.97969,    3.74336,    4.11725,    4.63007,    5.49768,    6.08738,    6.99323,    7.65307,    8.32950,    6.66260,    5.67879,    4.48326,    4.12765,    5.04904,    5.45831,    5.86582,    7.00003,    6.93604,    7.63841,    8.86530,    6.73468,    6.21781,    5.71358,    4.91231,    5.54536,    6.06927,    6.69714,    7.25546,    7.79128,    8.83760,    9.09183,    7.41139,    6.87444,    5.85150,    6.07245,    5.50734,    6.66435,    7.17681,    7.76632,    8.34496,    8.90111,    9.63815,    8.04956,    7.64247,    6.81406,    6.72227,    6.87724,    7.50079,    8.18540,    8.85608,    9.02513,    9.79967,   10.68150,    9.38231,    8.00004,    8.12097,    6.89211,    6.89117,    8.09478,    8.61705,    9.34025,   10.16863,   10.38486,   10.90012,    7.05939,    6.22342,    4.91330,    5.20695,    5.24919,    5.53948,    6.79029,    7.59295,    7.66363,    8.28244,    9.23756,    6.35466,    4.87945,    4.93230,    4.56045,    4.89626,    4.71244,    5.47979,    6.66455,    7.56514,    8.05578,    8.10883,    5.25643,    5.17825,    4.16513,    3.50695,    3.46038,    3.95872,    5.17736,    5.52739,    6.82570,    6.93404,    7.77257,    5.19983,    4.50118,    3.54318,    3.58816,    3.57505,    4.16931,    5.21190,    5.66511,    6.41266,    6.97889,    7.98338,    5.00511,    4.96488,    4.09482,    3.38596,    3.12467,    4.19221,    4.50570,    5.50003,    6.21229,    6.81461,    7.52089,    5.06220,    4.77840,    4.27714,    3.59816,    3.99634,    4.28351,    4.92475,    5.84853,    6.80822,    7.06953,    7.36271,    6.24811,    5.40740,    4.39186,    4.51009,    3.81435,    5.31131,    5.43192,    5.83672,    7.29614,    7.29497,    7.96098,    6.13176,    5.51681,    4.92236,    4.90963,    5.02611,    5.25765,    6.76579,    7.24905,    7.48313,    8.13244,    9.10595,    7.41106,    6.31690,    6.06003,    5.51183,    5.18957,    5.86773,    6.71342,    7.30747,    8.42109,    8.78919,    9.38936,    7.64994,    7.70587,    6.36881,    5.67612,    6.48667,    6.97913,    7.34944,    8.25670,    9.12666,   10.15640,   10.06010,    8.67514,    7.96548,    7.33517,    7.18190,    7.26590,    8.13710,    7.94614,    9.12261,    9.25238,    9.89265,   10.59068,    6.62248,    6.16200,    5.14823,    4.38425,    4.41148,    5.43029,    6.58529,    6.96254,    8.05482,    7.94448,    8.84822,    6.29188,    5.26197,    4.06697,    3.82188,    4.06231,    5.32962,    5.45614,    6.30060,    7.02047,    8.03753,    8.14909,    5.46093,    4.50705,    3.50162,    3.71934,    3.52818,    4.08404,    4.49883,    5.10391,    6.03429,    6.54826,    7.62028,    5.47239,    4.33299,    3.66634,    3.54126,    3.32594,    4.43946,    5.22420,    5.55151,    6.57361,    7.06377,    7.03698,    4.63384,    4.17290,    3.62691,    3.53449,    3.84000,    4.45796,    4.79264,    5.25197,    6.13414,    7.11212,    7.30073,    4.77827,    4.77055,    4.13088,    3.04167,    3.55509,    4.69975,    5.41660,    5.27449,    5.83835,    7.00558,    7.41333,    5.57882,    4.76698,    4.54537,    3.65026,    4.18001,    4.41284,    5.45622,    6.40184,    6.84937,    7.43771,    8.46057,    6.39431,    5.66490,    4.64858,    4.80546,    5.09605,    5.17229,    5.74843,    7.07551,    7.96896,    8.64851,    8.63119,    7.40351,    6.38779,    5.94849,    5.10026,    5.78272,    6.15430,    6.95296,    7.61106,    7.66617,    8.84968,    9.39605,    7.91901,    7.18683,    6.19662,    6.05039,    6.28826,    6.74032,    7.40780,    7.84178,    8.40569,    9.13729,    9.89733,    8.67312,    8.23767,    6.86931,    6.33131,    6.90508,    7.51140,    8.60368,    8.99240,    9.30860,   10.43723,   10.70175,    6.13130,    6.04119,    5.02785,    4.92106,    4.68665,    5.21608,    6.59491,    7.17484,    7.65974,    8.04679,    9.40529,    5.37149,    5.35407,    4.45672,    3.59389,    4.24730,    4.44888,    5.96150,    6.23546,    7.28895,    7.27428,    8.26402,    5.51646,    4.22346,    3.50052,    3.24933,    3.10556,    4.61927,    4.37739,    5.07488,    6.17530,    7.06279,    7.66338,    5.31872,    3.99911,    3.45996,    2.97825,    3.63427,    4.21137,    4.97317,    5.06745,    5.86423,    6.72474,    7.38050,    4.99355,    4.45449,    3.18712,    2.98805,    3.77466,    4.35511,    5.06677,    5.81782,    6.20399,    6.53987,    7.16486,    4.88229,    4.54164,    3.53398,    3.78357,    3.77848,    4.01215,    5.02207,    5.51837,    6.37922,    6.83879,    8.01226,    5.58013,    4.58403,    4.02982,    4.43108,    4.37504,    5.02450,    5.50525,    6.19835,    6.81148,    7.20028,    7.69431,    6.14706,    6.20697,    4.99980,    4.78064,    5.06952,    5.88675,    6.12753,    6.46532,    7.80319,    8.22496,    8.41591,    7.36131,    6.41203,    5.96263,    5.08112,    5.05222,    5.84953,    7.26890,    7.34668,    8.55682,    8.91937,    9.56476,    7.55013,    6.74261,    6.62092,    5.88718,    5.92702,    6.29458,    7.81232,    8.24517,    8.72092,    9.23759,    9.92744,    8.27843,    7.88144,    7.44679,    6.25348,    6.65881,    7.31030,    8.51727,    9.03720,    9.87032,    9.75437,   11.11186,    6.31094,    6.07694,    5.34222,    5.28947,    5.27883,    6.21419,    5.99540,    7.42951,    7.76176,    8.92210,    8.99740,    5.53042,    5.12789,    4.17631,    4.38273,    4.54868,    5.37267,    5.89089,    6.53187,    7.16252,    7.88937,    7.99745,    5.24851,    4.41155,    4.22144,    3.39827,    3.56525,    4.46850,    5.21551,    5.92094,    6.41620,    7.25922,    7.65267,    5.56111,    4.19985,    3.34556,    3.71404,    3.35535,    3.81939,    5.18699,    5.65059,    6.27639,    6.60507,    7.16287,    5.53351,    4.88010,    3.89906,    3.85348,    3.19683,    4.04681,    4.60290,    5.62944,    6.16642,    6.64344,    7.19016,    5.58863,    4.64547,    4.35303,    3.68603,    4.08939,    4.89292,    5.59070,    5.44822,    6.17133,    7.38940,    8.09645,    6.36890,    5.68184,    4.47711,    4.56905,    3.79736,    4.99097,    5.77432,    5.82555,    7.07446,    7.98652,    8.69075,    6.57138,    6.19377,    5.59360,    5.29669,    5.21808,    5.51760,    5.90601,    6.79074,    8.09713,    8.36159,    8.80418,    6.91408,    6.64260,    6.29816,    5.93852,    6.02933,    6.00853,    6.61303,    7.40887,    8.28736,    9.35142,   10.07318,    7.48201,    6.81793,    6.38134,    6.30455,    6.35932,    7.34902,    7.77413,    7.96026,    8.51172,    9.97335,    9.90896,    8.62461,    7.49690,    7.11798,    6.96486,    7.42341,    7.96638,    8.28718,    9.03453,    9.33209,   10.27968,   11.45468,    7.26563,    5.99882,    5.26241,    5.41759,    5.11686,    6.03716,    6.66149,    7.53378,    8.26613,    8.30766,    8.88051,    5.86581,    5.95872,    4.66118,    4.26643,    4.37928,    5.56468,    6.04994,    6.55787,    6.97392,    8.10876,    8.46363,    5.54896,    5.42142,    4.69771,    3.66567,    3.85944,    4.30988,    5.17195,    6.03798,    6.43105,    7.81142,    7.81028,    5.49232,    4.59501,    4.29910,    3.18462,    3.42335,    4.85369,    4.69540,    5.43438,    6.52252,    6.89354,    8.13233,    5.61790,    5.28105,    4.20305,    3.38972,    3.88655,    4.76391,    4.97641,    5.87315,    6.87836,    7.59358,    8.31916,    5.36369,    4.81371,    4.80089,    4.25741,    4.12162,    4.35165,    5.58352,    6.08709,    6.42471,    7.17703,    7.88680,    6.03129,    5.01171,    4.95024,    4.33264,    5.00982,    5.45641,    6.13132,    6.06057,    7.26076,    7.60476,    8.09908,    6.55569,    6.36019,    5.70024,    4.65047,    5.18216,    5.41944,    6.79515,    6.90753,    8.22318,    8.09395,    8.98071,    7.26370,    6.88911,    6.53150,    6.06867,    5.40919,    6.61738,    7.00037,    7.80755,    8.94083,    9.06221,    9.57719,    8.49118,    7.15688,    6.85342,    6.82929,    6.68034,    7.20322,    7.43460,    8.28425,    8.87696,    9.90997,   10.71619,    8.85149,    7.87988,    7.96574,    7.40157,    6.85766,    8.21062,    8.56694,    9.55384,   10.07141,   10.70559,   11.13272,    7.25516,    6.99601,    6.27041,    5.74369,    5.99925,    6.61733,    7.22422,    7.65232,    8.18771,    9.31913,   10.25201,    6.77500,    5.87299,    5.61497,    5.34829,    4.94837,    5.44391,    6.55722,    6.62041,    7.64043,    7.99521,    9.25411,    6.43300,    5.79589,    4.59331,    3.91390,    4.61202,    4.78426,    5.64791,    6.84438,    6.94575,    7.34044,    8.72211,    6.20660,    5.31626,    4.36214,    4.34121,    4.46543,    5.21805,    5.55532,    6.12285,    7.36407,    7.13724,    8.43614,    5.84588,    5.60268,    4.94830,    4.03420,    4.16327,    5.22943,    5.47569,    6.20399,    7.30876,    7.42009,    8.01810,    6.19063,    5.02848,    5.09410,    4.10503,    4.88819,    5.17179,    6.31187,    6.73046,    6.71517,    8.33690,    8.98367,    6.11399,    5.99004,    5.41294,    4.49910,    4.49417,    5.16912,    6.24512,    7.11950,    7.53090,    8.57020,    9.49076,    7.42456,    6.77099,    6.30021,    5.81430,    5.87457,    6.03383,    7.32649,    7.58192,    8.31540,    9.10758,    9.72049,    7.70624,    7.05296,    6.23885,    6.09888,    6.38972,    7.15295,    7.26598,    7.94747,    8.79909,    9.63054,   10.21812,    9.05202,    8.26772,    7.05734,    6.40210,    6.97657,    7.73226,    8.07136,    8.69152,   10.13639,   10.10578,   11.03121,    9.14414,    8.45831,    7.48650,    7.18451,    7.64347,    8.44723,    8.76352,    9.94959,    9.94575,   10.91314,   11.64069,    8.28308,    7.27744,    6.29554,    5.75323,    6.43776,    6.98259,    7.95862,    8.02277,    9.32121,    9.84747,   10.73401,    7.46297,    6.26463,    6.16094,    5.35348,    5.77750,    6.72351,    7.14227,    7.62109,    8.23061,    9.02314,    9.58476,    6.48847,    6.38447,    5.14181,    5.07391,    4.56413,    5.72335,    6.41337,    7.04900,    8.01756,    8.40263,    9.05034,    6.39624,    6.00076,    5.42709,    4.12712,    5.09096,    5.42118,    5.98034,    7.09417,    7.82498,    7.92013,    8.88578,    6.64989,    5.73621,    5.26131,    4.15421,    5.23072,    5.83775,    6.61972,    6.40808,    7.13311,    8.60052,    8.79266,    6.16796,    5.51624,    5.42285,    4.69252,    5.14749,    5.94564,    6.50497,    7.40155,    7.86901,    8.63067,    8.97307,    6.94797,    6.19211,    5.65625,    5.57248,    5.47774,    6.00957,    6.57672,    7.72407,    8.62283,    8.61125,    9.39819,    7.30842,    7.13548,    5.98406,    5.83100,    6.31118,    6.74756,    7.60400,    8.11926,    9.06977,    9.58272,    9.70307,    8.84621,    7.59871,    6.74654,    6.42611,    6.43864,    7.45907,    7.96237,    8.89353,    9.60579,   10.24794,   11.17045,    9.10455,    8.77476,    7.78706,    7.06172,    7.01587,    8.44521,    8.38660,    9.13108,    9.97380,   10.83204,   11.07585,   10.28339,    9.34327,    8.32340,    8.14155,    7.91796,    8.89945,    9.40687,   10.12608,   11.13137,   11.70371,   11.73649,    8.28039,    8.30922,    7.29090,    7.19163,    6.45673,    8.11136,    8.10530,    9.40975,    9.78310,    9.97835,   11.16658,    8.22994,    6.77130,    6.66160,    6.17775,    6.06818,    6.74383,    7.61600,    8.41760,    9.21259,    9.36633,   10.54526,    7.05821,    6.23133,    6.34785,    5.09461,    5.66269,    6.68132,    6.90581,    8.05841,    8.76989,    8.91805,    9.58037,    7.08265,    6.76007,    5.52746,    5.52113,    5.79520,    6.10163,    6.60880,    7.62168,    7.96690,    8.43393,    8.98972,    6.74550,    6.84626,    5.26839,    4.93577,    5.59374,    6.38398,    7.14417,    7.67605,    8.12706,    8.36827,    9.79716,    7.55257,    6.26120,    5.99282,    6.04051,    5.29555,    6.70334,    6.85948,    8.06969,    7.97970,    8.60753,    9.46175,    8.21890,    7.29187,    6.33787,    5.64777,    6.40851,    7.10557,    7.10677,    8.21778,    8.69983,    9.51281,   10.16077,    8.62397,    7.49458,    7.39079,    6.32418,    7.00370,    7.32519,    8.02988,    8.52323,    9.30163,   10.08879,   11.25766,    9.32721,    8.30509,    7.54249,    7.23023,    7.94806,    8.51172,    8.78312,    9.87132,   10.13202,   11.23649,   11.97090,    9.89598,    9.55967,    8.75919,    8.41391,    8.05919,    8.75678,    9.36990,    9.71001,   10.78511,   11.25404,   12.57083,   10.93299,    9.52843,    9.59140,    8.57751,    9.22801,    9.92448,   10.35240,   11.18401,   11.90254,   12.55243,   12.44205,    9.37010,    9.03166,    8.19348,    7.59862,    7.83377,    8.04345,    9.26704,   10.11295,    9.85914,   11.06051,   11.88691,    8.39578,    7.82100,    7.02624,    6.71696,    7.27338,    7.60015,    8.02455,    8.87717,    9.20911,    9.83862,   11.31957,    7.85826,    6.87347,    6.53884,    5.99019,    6.66195,    6.90270,    7.19108,    8.69279,    8.68111,    9.40835,   10.73904,    8.17938,    7.10535,    6.20760,    6.40704,    5.75948,    7.19469,    7.82215,    8.05949,    8.72121,    9.84234,   10.15351,    7.35330,    7.21027,    6.38736,    6.35303,    5.92936,    6.66404,    7.85510,    7.89709,    8.56491,    9.18837,   10.19700,    8.23361,    7.77812,    6.99266,    5.88155,    6.65754,    7.02790,    8.15055,    8.21389,    9.20495,    9.26698,   10.70574,    8.80219,    8.05280,    7.41475,    6.55806,    7.10107,    7.49558,    8.24945,    8.60011,    9.30574,   10.12667,   11.29310,    9.44930,    8.57032,    8.02522,    7.74798,    7.46884,    8.06872,    8.52789,    9.85040,    9.80704,   10.62138,   11.70439,   10.25642,    9.20054,    8.40430,    7.65160,    7.75639,    9.28544,    9.75477,    9.96484,   11.09775,   11.10942,   12.22974,   10.13918,    9.92499,    8.74978,    8.56344,    8.95902,    9.51338,   10.47871,   11.15770,   11.05041,   12.62711,   13.32108,   10.68936,   10.00686,    9.58011,    9.77466,    9.60712,   10.43850,   10.49117,   11.60838,   12.29384,   13.36797,   13.59658])
    # for j in range(len(rewards)):
    #     rewards[j] = 0
    for i in range(len(trajectories[0])):
        # print "traj is ", trajectories[0][i]
        index_vals[i] = env_obj.get_state_val_index(trajectories[0][i])
    # for _, ele in enumerate(index_vals):
    #     if ele == index_val:
    #         rewards[int(ele)] = 100
    #     else:
    #         rewards[int(ele)] = 1
    env_obj.rewards = -1*rewards
    print "rewards is ", rewards
    # initialize V(s)
    V = {}
    for state in env_obj.states:
        V[state] = 0

    # Reinitialize policy
    policy = {}
    for state in env_obj.states:
        policy[state] = [key for key in env_obj.action_space]
    for i in range(3):
        V, policy = env_obj.iterateValues(V, policy, 0.001)

    # print(V)
    print(policy)

    policy_arr = np.zeros(len(states))
    for j, ele in policy.items():
        policy_arr[j] = ele

    filename = "/home/vvarier/dvrk_automated_suturing/iros2020/suturing_grid3d_size11_data1/grid11_parallel/policy_grid11.txt"
    # filename = "/home/vignesh/Desktop/policy_grid11.txt"

    file_open = open(filename, 'a')
    savetxt(file_open, policy_arr, delimiter=',', fmt="%10.5f", newline=", ")
    file_open.write("\n \n \n \n")
    file_open.close()


'''
class RobotStateUtils(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, grid_size, discount, terminal_state_val_from_trajectory):
        super(RobotStateUtils, self).__init__(max_workers=8)
        # Model here means the 3D cube being created
        # linspace limit values: limit_values_pos = [[-0.009, -0.003], [0.003, 007], [-0.014, -0.008]]
        # Creates the model state space based on the maximum and minimum values of the dataset provided by the user
        # It is for created a 3D cube with 3 values specifying each cube node
        # The value 11 etc decides how sparse the mesh size of the cube would be
        self.grid_size = grid_size
        self.lin_space_limits = np.linspace(-5, 5, self.grid_size, dtype='float32')
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
        self.n_states = grid_size**3
        self.n_actions = 27
        self.gamma = discount

        self.rewards = []
        self.values_tmp = np.zeros([self.n_states])



    def create_state_space_model_func(self):
        # Creates the state space of the robot based on the values initialized for linspace by the user
        # print "Creating State space "
        state_set = []
        for i_val in self.lin_space_limits:
            for j_val in self.lin_space_limits:
                for k_val in self.lin_space_limits:
                    # Rounding state values so that the values of the model, dont take in too many floating points
                    state_set.append([round(i_val, 1), round(j_val, 1), round(k_val, 1)])
        # Assigning the dictionary keys
        for i in range(len(state_set)):
            state_dict = {i: state_set[i]}
            self.states.update(state_dict)

        return self.states

    def create_action_set_func(self):
        # Creates the action space required for the robot. It is defined by the user beforehand itself
        action_set = []
        for pos_x in [-1, 0, 1]:
            for pos_y in [-1, 0, 1]:
                for pos_z in [-1, 0, 1]:
                    action_set.append([pos_x, pos_y, pos_z])
        # Assigning the dictionary keys
        for i in range(len(action_set)):
            action_dict = {i: action_set[i]}
            self.action_space.update(action_dict)

        return self.action_space

    def get_state_val_index(self, state_val):
        index_val = abs((state_val[0] + 5) * pow(self.grid_size, 2)) + abs((state_val[1] + 5) * pow(self.grid_size, 1)) + \
                    abs((state_val[2] + 5))
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
        elif (x % self.grid_size for x in old_state) == 0 and (y % self.grid_size for y in new_state) == self.grid_size - 1:
            return True
        elif (x % self.grid_size for x in old_state) == self.grid_size - 1 and (y % self.grid_size for y in new_state) == 0:
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
'''
# def reward_func(self, end_pos_x, end_pos_y, end_pos_z, alpha):
#     # Creates list of all the features being considered
#
#     # reward = -1
#     if self.is_terminal_state([end_pos_x, end_pos_y, end_pos_z]):
#         reward = 0
#     else:
#         reward = -1
#
#     return reward, 1, 2
'''
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
            resulting_state.append(round(self.states[curr_state][i] + self.action_space[action][i], 1))

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

    # def get_transition_states_and_probs(self, curr_state, action):
    #
    #     if self.is_terminal_state(curr_state):
    #         return [(curr_state, 1)]
    #     resulting_state = []
    #     if self.trans_prob == 1:
    #         for i in range(0, self.n_params_for_state):
    #             resulting_state.append(round(self.states[curr_state][i] + self.action_space[action][i], 1))
    #         resulting_state_index = self.get_state_val_index(resulting_state)
    #
    #         if not self.off_grid_move(resulting_state, self.states[curr_state]):
    #             # return resulting_state, reward, self.is_terminal_state(resulting_state_index), None
    #             return [(resulting_state_index, 1)]
    #         else:
    #             # if the state is invalid, stay in the current state
    #             return [(curr_state, 1)]

    # def get_transition_mat_deterministic(self):
    #
    #     self.n_actions = len(self.action_space)
    #     for si in range(self.n_states):
    #         for a in range(self.n_actions):
    #             probabilities = self.get_transition_states_and_probs(si, a)
    #
    #             for next_pos, prob in probabilities:
    #                 # sj = self.get_state_val_index(posj)
    #                 sj = int(next_pos)
    #                 # Prob of si to sj given action a
    #                 prob = int(prob)
    #                 self.P_a[si, a, sj] = prob
    #     return self.P_a

    def calc_value_for_state(self, s):
        value = max([sum([(self.rewards[s] + self.gamma * self.values_tmp[s1]) for s1 in range(self.n_states)])])
        return value, s

    def value_iteration(self, rewards, error=0.001):
        # Initialize the value function
        # t_complete_func = TicToc()
        # t_complete_func.tic()
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
                print "\nvalues is ", values[s]
'''
            # for s in range(self.n_states):
            #     values[s] = max(
            #         [sum([P_a[s, a, s1] * (rewards[s] + self.gamma * values_tmp[s1])
            #               for s1 in range(self.n_states)])
            #          for a in range(self.n_actions)])
'''
            # t_value.toc('Value function section took')
                # print "values ", values[s]
            if max([abs(values[s] - self.values_tmp[s]) for s in range(self.n_states)]) < error:
                break
        # generate deterministic policy
        policy = np.zeros([self.n_states])
        for s in range(self.n_states):
            policy[s] = np.argmax([sum([(self.rewards[s] + self.gamma * values[s1])
                                        for s1 in range(self.n_states)])])

        # t_complete_func.toc('Complete function section took')
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
                mu[s, t + 1] = sum([mu[pre_s, t] for pre_s in range(self.n_states)])
        p = np.sum(mu, 1)
        return p



if __name__ == '__main__':
    weights = np.array([[1, 1, 0]])
    # term_state = np.random.randint(0, grid_size ** 3)]
    env_obj = RobotStateUtils(11, weights, 0.9)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    print "State space created is ", states
    print "actions is ", action
    index_val = env_obj.get_state_val_index([5, -5, 0])
    print "index val is ", index_val
'''
'''
    # Robot Object called
    # Pass the gridsize required
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



















