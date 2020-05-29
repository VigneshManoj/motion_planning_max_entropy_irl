# dict = {0: "hi", 1: "vignesh", 2: "vig"}
# print dict.items()
import numpy as np
filename = "/home/vignesh/trial.txt"
# a = np.array([4.91655950e-03, 5.01454057e-01, 4.06537812e+00, 8.05908637e-0])
# np.savetxt(filename, a, fmt='%f', delimiter='\t')


# save numpy array as csv file
from numpy import savetxt

# define data
data = np.around(np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), decimals=1)
# save to csv file
print data
savetxt(filename, data, delimiter=',', fmt="%10.4f")