import numpy as np
from matplotlib import pyplot as plt
from robot_state_utils import RobotStateUtils
from numpy import savetxt
import pandas as pd

if __name__ == '__main__':
    array_val = np.array([1, 2, 5, 30, 137, 5, 86, 21, 94, 29, 11, 46, 8, 61, 49, 47, 6, 66, 16, 4])
    print("Average is ", np.average(array_val))