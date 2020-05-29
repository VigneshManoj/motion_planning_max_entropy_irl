import numpy as np
from matplotlib import pyplot as plt
from robot_state_utils import RobotStateUtils
from numpy import savetxt
from robot_markov_model import RobotMarkovModel


if __name__ == '__main__':
    filename = "/home/vignesh/Desktop/individual_trials/version4/data1/policy_grid11.txt"
    # term_state = np.random.randint(0, grid_size ** 3)]
    goal = np.array([0.005, 0.055, -0.125])
    env_obj = RobotStateUtils(11, 0.01, goal)
    mdp_obj = RobotMarkovModel()
    trajectories = mdp_obj.generate_trajectories()
    index_vals = np.zeros(len(trajectories[0]))
    for i in range(len(trajectories[0])):
        # print "traj is ", trajectories[0][i]
        index_vals[i] = env_obj.get_state_val_index(trajectories[0][i])
    # print index_vals
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    rewards = np.zeros(len(states))
    index = env_obj.get_state_val_index(goal)
    for _, ele in enumerate(index_vals):
        if ele == index:
            rewards[int(ele)] = 10
        else:
            rewards[int(ele)] = 1

    print "State space created is ", states
    print "actions is ", action
    print "highest reward ", states[np.argmax(rewards)], np.max(rewards), rewards[795], np.argmax(rewards)
    print index, states[index-1], states[index], states[index+1]
    # map magic squares to their connecting square
    # rewards[904] = 25
    # P_a = env_obj.get_transition_mat_deterministic()
    policy = env_obj.value_iteration(rewards)
    # print("Policy is ", policy.reshape((121, 11)))
    file_open = open(filename, 'a')
    savetxt(file_open, policy, delimiter=',', fmt="%10.5f", newline=", ")
    file_open.write("\n \n \n \n")
    file_open.close()






    # model hyperparameters
    # ALPHA = 0.1
    # GAMMA = 1.0
    # EPS = 1.0
    #
    # Q = {}
    # for state in env_obj.stateSpacePlus:
    #     for action in env_obj.possibleActions:
    #         Q[state, action] = 0
    #
    # numGames = 50000
    # totalRewards = np.zeros(numGames)
    # for i in range(numGames):
    #     if i % 5000 == 0:
    #         print('starting game ', i)
    #     done = False
    #     epRewards = 0
    #     observation = env_obj.reset()
    #     while not done:
    #         rand = np.random.random()
    #         action = env_obj.maxAction(Q,observation, env_obj.possibleActions) if rand < (1-EPS) \
    #                                                 else env_obj.actionSpaceSample()
    #         observation_, reward, done, info = env_obj.step(action)
    #         epRewards += reward
    #
    #         action_ = env_obj.maxAction(Q, observation_, env_obj.possibleActions)
    #         Q[observation,action] = Q[observation,action] + ALPHA*(reward + \
    #                     GAMMA*Q[observation_,action_] - Q[observation,action])
    #         observation = observation_
    #     if EPS - 2 / numGames > 0:
    #         EPS -= 2 / numGames
    #     else:
    #         EPS = 0
    #     totalRewards[i] = epRewards
    #
    # plt.plot(totalRewards)
    # plt.show()