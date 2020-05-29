import numpy as np
from numpy import genfromtxt

if __name__ == "__main__":

    file = "/home/vignesh/PycharmProjects/motion_planning_max_entropy_irl/RL_3d_gridsize_11/user_traj.csv"

    my_data = genfromtxt(file, delimiter=',')
    print(my_data)
    np.save('user_traj', my_data)