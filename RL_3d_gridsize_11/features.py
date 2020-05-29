import numpy as np


# actions = np.load('/home/vignesh/Downloads/expert_cartpole/actions.npy')
# ep_ret = np.load('/home/vignesh/Downloads/expert_cartpole/episode_returns.npy')
# epi_start = np.load('/home/vignesh/Downloads/expert_cartpole/episode_starts.npy')
# obs = np.load('/home/vignesh/Downloads/expert_cartpole/obs.npy')
# rewards = np.load('/home/vignesh/Downloads/expert_cartpole/rewards.npy')
# output = np.load('/home/vignesh/PycharmProjects/motion_planning_max_entropy_irl/RL_3d_gridsize_11/different_policies/output1.npy')
# print(actions.shape)
# print(ep_ret.shape, ep_ret[0])
# print(epi_start.shape, epi_start[0])
# print(obs.shape)
# print(rewards.shape)
# for i in range(len(epi_start)):
#     print(epi_start[i])
# dic = {'actions': np.array([1, 2, 3]), 'rewards': np.array([4, 5, 6])}
# print(dic)
# np.savez('model.npz', **dic)
# print(output.reshape((11, 3)))
# np.savetxt('/home/vignesh/Thesis_Suture_data/trial2/SD/test.csv', output.reshape((11, 3)), delimiter=',')
actions = np.load('/home/vignesh/Thesis_Suture_data/trial2/ambf_data/actions.npy')
episode_returns = np.load('/home/vignesh/Thesis_Suture_data/trial2/ambf_data/episode_returns.npy')
episode_starts = np.load('/home/vignesh/Thesis_Suture_data/trial2/ambf_data/episode_starts.npy')
observations = np.load('/home/vignesh/Thesis_Suture_data/trial2/ambf_data/obs.npy')
rewards = np.load('/home/vignesh/Thesis_Suture_data/trial2/ambf_data/rewards.npy')
file_dir = "/home/vignesh/Thesis_Suture_data/trial2/ambf_data/"

numpy_dict = {
    'actions': actions,
    'obs': observations,
    'rewards': rewards,
    'episode_returns': episode_returns,
    'episode_starts': episode_starts
}
# print(dic)
np.savez(file_dir + 'expert_psm_data.npz', **numpy_dict)