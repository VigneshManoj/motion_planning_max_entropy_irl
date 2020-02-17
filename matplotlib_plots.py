import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.preprocessing import normalize
np.set_printoptions(precision=4, linewidth=200)

trajectories_z0 = [[0, 0, 0] for i in range(121)]
default_points = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
i = 0
for y in default_points:
  for x in default_points:
    trajectories_z0[i] = [x, y, 0.]
    i += 1

weights = np.array([4.91655950e-03, 8.47214258e-01, 9.00809071e-02, 3.07317903e-01,
 8.31000653e-01, 4.10764270e-01, 2.04060226e+00, 6.52020263e-01,
 3.79428710e-01, 7.92978516e-02, 1.06815971e-01, 7.58752997e-01,
 8.20599087e-01, 3.21564630e-01, 9.46170105e-01, 1.42133597e-02,
 3.32595180e-01, 4.45346761e-01, 6.35684337e-01, 7.85756779e-01,
 2.10709616e-01, 8.14521162e-01, 2.25017831e-01 ,1.74620724e-01,
 6.39265747e-02, 3.05675356e-01, 3.20903188e-01, 6.70772303e-02,
 5.74115953e-01, 1.33073344e-01, 7.90480075e-01, 9.55374756e-02,
 5.68879460e-01, 8.50135036e-02, 1.05296049e-01, 2.33878797e-01,
 6.57948372e-01, 3.29637775e-01, 8.75917867e-01, 7.07005408e-01,
 8.59177874e-01, 8.23875110e-02, 7.00787227e-02, 2.81862802e-01,
 2.39227396e-01, 4.22346823e-01, 2.02522469e-01, 4.69581783e-01,
 1.89937792e-01, 8.83255971e-01, 2.80882149e+00, 1.11794244e-01,
 6.98814887e-01, 8.93339083e-01, 7.80497899e-01, 4.65745322e-01,
 3.68488115e-01, 7.84075783e-01, 7.09096454e-01, 4.68195475e-01,
 6.09130411e-01, 3.65545717e-02 ,1.53833696e+00, 5.59364365e-01,
 5.90444150e-01, 5.18695040e-02, 8.27224242e-01, 2.57007598e-01,
 5.58997695e-01, 5.73757870e-01, 1.29341287e-01, 5.71707702e-01,
 1.29860570e-01, 5.26232459e-01, 1.49905389e+00, 3.23685916e-01,
 4.66216946e-02, 6.25790760e-01, 5.84025374e-01, 6.91718414e-01,
 6.36657129e-01, 9.63193267e-01, 9.56778741e-01, 5.35653476e-01,
 1.67028387e+00, 6.29770878e-01, 2.66502999e-01, 3.07614883e-01,
 5.09487866e-01, 5.45739571e-01 ,4.69653339e-01, 7.64258768e-01,
 7.85221374e-01, 5.47740287e-01, 1.96223598e+00, 8.26223515e-01,
 8.38864054e-01, 3.64932107e-01, 1.19796080e-01, 5.63416979e-01,
 2.56821009e-01, 6.38387957e-01, 6.16690720e-01, 3.88490536e-03,
 5.71117436e-01, 9.38111927e-01, 1.54737366e+00, 5.69436498e-02,
 1.88873573e-01, 2.26993372e-01, 5.37305095e-01, 7.21024730e-01,
 1.24971397e-01, 9.12882325e-01, 2.36292349e-02, 9.14253269e-01,
 6.43959191e-01, 5.01454057e-01, 4.06537812e+00, 8.05908637e-01,
 8.28773151e-01])

Mat = weights.reshape((11, 11), order='F')
print "mat ", Mat
minMaxMat = np.zeros_like(Mat)
for i, col in enumerate(Mat.T):
    mx = np.max(col)
    for j, row in enumerate(col):
        if Mat[j][i] == mx:
            minMaxMat[i][j] = 1
        else:
            minMaxMat[i][j] = 0
print minMaxMat.T
minMaxMat = minMaxMat.T

mFlat = minMaxMat.flatten()

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(121):
  trajectories_z0[i][2] = mFlat[i]
  ax.scatter(
    trajectories_z0[i][0],
    trajectories_z0[i][1],
    s=trajectories_z0[i][2] * 100,
    marker="o")
ax.legend(loc=8, framealpha=1, fontsize=8)
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
plt.title('Distribution of Reward Function for a custom trajectory')
plt.show()
fig.savefig("reward_plot")