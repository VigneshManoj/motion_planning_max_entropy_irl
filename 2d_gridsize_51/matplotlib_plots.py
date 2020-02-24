import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from sklearn.preprocessing import normalize
# np.set_printoptions(precision=4, linewidth=200)

trajectories_z0 = [[0, 0, 0] for i in range(2601)]
default_points = np.array([-25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, -11, -10, -9, -8, -7,
                           -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                           19, 20, 21, 22, 23, 24, 25])
i = 0
for y in default_points:
  for x in default_points:
    trajectories_z0[i] = [x, y, 0.]
    i += 1
#
# weights = np.array([4.91655950e-03, 8.47214258e-01, 9.00809071e-02, 3.07317903e-01,
#  8.31000653e-01, 4.10764270e-01, 2.04060226e+00, 6.52020263e-01,
#  3.79428710e-01, 7.92978516e-02, 1.06815971e-01, 7.58752997e-01,
#  8.20599087e-01, 3.21564630e-01, 9.46170105e-01, 1.42133597e-02,
#  3.32595180e-01, 4.45346761e-01, 6.35684337e-01, 7.85756779e-01,
#  2.10709616e-01, 8.14521162e-01, 2.25017831e-01 ,1.74620724e-01,
#  6.39265747e-02, 3.05675356e-01, 3.20903188e-01, 6.70772303e-02,
#  5.74115953e-01, 1.33073344e-01, 7.90480075e-01, 9.55374756e-02,
#  5.68879460e-01, 8.50135036e-02, 1.05296049e-01, 2.33878797e-01,
#  6.57948372e-01, 3.29637775e-01, 8.75917867e-01, 7.07005408e-01,
#  8.59177874e-01, 8.23875110e-02, 7.00787227e-02, 2.81862802e-01,
#  2.39227396e-01, 4.22346823e-01, 2.02522469e-01, 4.69581783e-01,
#  1.89937792e-01, 8.83255971e-01, 2.80882149e+00, 1.11794244e-01,
#  6.98814887e-01, 8.93339083e-01, 7.80497899e-01, 4.65745322e-01,
#  3.68488115e-01, 7.84075783e-01, 7.09096454e-01, 4.68195475e-01,
#  6.09130411e-01, 3.65545717e-02 ,1.53833696e+00, 5.59364365e-01,
#  5.90444150e-01, 5.18695040e-02, 8.27224242e-01, 2.57007598e-01,
#  5.58997695e-01, 5.73757870e-01, 1.29341287e-01, 5.71707702e-01,
#  1.29860570e-01, 5.26232459e-01, 1.49905389e+00, 3.23685916e-01,
#  4.66216946e-02, 6.25790760e-01, 5.84025374e-01, 6.91718414e-01,
#  6.36657129e-01, 9.63193267e-01, 9.56778741e-01, 5.35653476e-01,
#  1.67028387e+00, 6.29770878e-01, 2.66502999e-01, 3.07614883e-01,
#  5.09487866e-01, 5.45739571e-01 ,4.69653339e-01, 7.64258768e-01,
#  7.85221374e-01, 5.47740287e-01, 1.96223598e+00, 8.26223515e-01,
#  8.38864054e-01, 3.64932107e-01, 1.19796080e-01, 5.63416979e-01,
#  2.56821009e-01, 6.38387957e-01, 6.16690720e-01, 3.88490536e-03,
#  5.71117436e-01, 9.38111927e-01, 1.54737366e+00, 5.69436498e-02,
#  1.88873573e-01, 2.26993372e-01, 5.37305095e-01, 7.21024730e-01,
#  1.24971397e-01, 9.12882325e-01, 2.36292349e-02, 9.14253269e-01,
#  6.43959191e-01, 5.01454057e-01, 4.06537812e+00, 8.05908637e-01,
#  8.28773151e-01])
weights = np.array([0.00681,    0.36204,    0.29746,    0.36070,    0.76019,    0.28134,    4.65450,    0.67014,    0.09923,    0.80228,    0.11713,    0.03748,    0.97144,    0.84591,    0.12396,    0.13637,    0.58259,    0.48002,    0.23707,    0.50334,    0.66699,    0.87235,    0.55586,    0.42865,    0.43476,    0.79698,    0.88531,    0.72275,    0.23863,    0.03914,    0.93775,    0.38184,    0.25236,    0.77684,    0.32800,    0.20167,    0.14241,    0.46803,    0.78876,    0.09008,    0.63356,    0.90908,    0.79762,    0.66262,    0.57921,    0.56562,    0.73338,    0.53504,    0.27451,    0.07405,    0.31674,    0.44827,    0.64385,    0.18659,    0.91992,    0.26779,    0.09557,    0.04761,    6.24905,    0.40307,    0.06145,    0.85372,    0.20693,    0.40502,    0.86463,    0.77398,    0.34646,    0.07811,    0.96214,    0.72109,    0.01349,    0.74229,    0.83457,    0.08624,    0.33648,    0.66611,    0.63419,    0.07159,    0.00740,    0.83744,    0.76377,    0.41046,    0.70105,    0.99411,    0.56444,    0.69787,    0.36129,    0.32174,    0.48405,    0.29227,    0.03440,    0.90881,    0.43224,    0.53352,    0.84825,    0.01212,    0.68154,    0.48235,    0.20803,    0.52212,    0.90154,    0.89154,    0.16492,    0.19742,    0.34613,    0.89369,    0.32735,    0.59075,    0.97978,    0.31764,    5.81532,    0.01909,    0.85025,    0.24500,    0.25462,    0.43409,    0.15468,    0.53457,    0.76808,    0.93087,    0.13220,    0.26515,    0.89479,    0.97537,    0.67695,    0.13275,    0.68825,    0.37321,    0.99601,    0.83779,    0.57854,    0.75063,    0.26972,    0.84987,    0.47853,    0.57492,    0.82194,    0.30592,    0.94422,    0.93347,    0.13157,    0.35538,    0.30597,    0.27791,    0.62668,    0.26371,    0.82074,    0.77551,    0.31492,    0.14709,    0.69527,    0.99285,    0.37037,    0.08095,    0.06095,    0.82395,    0.22253,    0.48565,    0.84008,    0.22065,    6.08960,    0.81698,    0.99271,    0.67616,    0.08975,    0.47708,    0.10508,    0.14178,    0.08845,    0.13576,    0.85923,    0.09523,    0.13842,    0.93885,    0.82455,    0.47392,    0.66072,    0.82391,    0.80262,    0.16909,    0.76309,    0.57556,    0.08688,    0.53896,    0.93075,    0.63011,    0.81390,    0.32340,    0.26435,    0.75409,    0.58434,    0.95948,    0.81344,    0.70025,    0.11238,    0.05242,    0.93124,    0.77594,    0.11162,    0.92613,    0.73262,    0.55194,    0.54453,    0.42508,    0.99249,    0.16278,    0.42339,    0.97153,    0.65513,    0.22311,    6.67753,    0.29618,    0.78244,    0.67907,    0.48317,    0.81364,    0.16226,    0.56356,    0.48773,    0.41990,    0.08366,    0.90015,    0.44368,    0.79873,    0.83378,    0.89049,    0.43470,    0.13404,    0.80810,    0.82218,    0.87987,    0.64564,    0.19336,    0.36031,    0.59952,    0.46914,    0.43306,    0.14289,    0.86655,    0.32229,    0.62234,    0.21857,    0.58771,    0.43085,    0.00548,    0.42261,    0.91351,    0.13905,    0.31840,    0.89080,    0.59372,    0.74282,    0.23120,    0.62325,    0.81816,    0.59343,    0.61485,    0.95020,    0.10713,    0.80501,    0.67717,    0.31407,    6.33671,    0.49747,    0.77248,    0.96815,    0.30228,    0.65480,    0.26247,    0.69792,    0.77536,    0.33782,    0.28067,    0.48640,    0.56107,    0.62366,    0.99540,    0.31559,    0.91062,    0.12632,    0.15713,    0.30965,    0.01049,    0.34253,    0.50548,    0.72747,    0.41770,    0.37735,    0.74059,    0.78084,    0.63126,    0.82190,    0.35091,    0.74585,    0.09384,    0.94515,    0.55784,    0.43283,    0.10851,    0.17868,    0.13867,    0.21661,    0.75065,    0.61273,    0.86549,    0.03826,    0.43694,    0.99759,    0.36443,    0.53671,    0.16256,    0.37367,    0.26298,    0.06007,    5.95168,    0.25651,    0.64037,    0.98940,    0.27554,    0.23577,    0.67874,    0.64461,    0.08949,    0.33580,    0.43926,    0.40379,    0.69958,    0.44360,    0.81685,    0.36615,    0.53597,    0.07664,    0.35792,    0.92032,    0.80972,    0.89924,    0.61988,    0.61426,    0.30815,    0.04006,    0.22056,    0.71396,    0.82409,    0.42668,    0.90350,    0.22101,    0.47752,    0.40676,    0.69806,    0.11704,    0.53767,    0.68735,    0.69025,    0.61434,    0.85814,    0.33991,    0.29310,    0.08722,    0.20427,    0.87593,    0.91528,    0.05503,    0.21942,    0.83941,    5.86048,    0.23626,    0.81408,    0.35423,    0.84531,    0.58911,    0.61307,    0.26682,    0.88799,    0.35364,    0.78950,    0.72123,    0.80275,    0.46840,    0.89401,    0.27591,    0.02209,    0.76239,    0.23615,    0.29521,    0.25497,    0.87818,    0.00147,    0.16061,    0.92127,    0.29596,    0.69924,    0.99198,    0.97143,    0.69141,    0.52627,    0.70164,    0.29459,    0.30832,    0.73770,    0.57083,    0.10183,    0.39956,    0.77354,    0.27083,    0.68404,    0.95089,    0.08308,    0.77741,    0.51577,    0.07490,    6.07287,    7.88199,    0.52129,    0.14603,    6.63681,    0.96588,    0.76453,    0.18729,    0.03259,    0.49032,    0.03649,    0.99197,    0.40098,    0.93280,    0.50460,    0.68003,    0.70479,    0.29803,    0.60386,    0.39123,    0.09426,    0.54248,    0.57112,    0.58264,    0.72541,    0.03239,    0.38449,    0.83732,    0.22318,    0.73153,    0.70904,    0.92027,    0.70420,    0.22018,    0.62308,    0.16604,    0.98770,    0.22194,    0.33621,    0.22804,    0.73532,    0.14659,    0.70725,    0.69015,    0.78792,    0.36589,    0.99546,    0.63536,    0.57429,    0.05854,    0.99755,    0.16403,    6.23995,    0.22399,    0.61083,    0.77707,    5.84825,    0.28387,    0.46924,    0.81595,    0.44636,    0.73308,    0.19596,    0.53725,    0.53145,    0.01424,    0.66856,    0.50846,    0.31705,    0.34770,    0.57459,    0.71696,    0.94416,    0.46623,    0.04540,    0.27136,    0.75033,    0.06391,    0.39179,    0.03941,    0.27972,    0.73531,    0.77902,    0.62365,    0.53154,    0.83230,    0.14501,    0.54193,    0.53176,    0.64179,    0.85092,    0.01056,    0.65060,    0.01813,    0.93769,    0.73069,    0.46052,    0.41993,    0.89646,    0.78994,    0.91531,    0.88400,    0.01039,    0.96432,    6.20140,    8.46015,    6.14982,    5.81385,    8.48814,    0.60998,    0.79768,    0.14850,    0.88211,    0.38298,    0.15257,    0.77534,    0.66273,    0.40783,    0.35776,    0.54594,    0.13716,    0.01083,    0.56610,    0.43880,    0.27492,    0.69838,    0.76439,    0.66825,    0.25643,    0.38878,    0.60016,    0.37128,    0.43777,    0.29886,    0.87948,    0.50872,    0.94446,    0.57953,    0.62357,    0.04000,    0.04467,    0.78937,    0.62580,    0.12696,    0.33910,    0.07774,    0.56325,    0.38416,    0.01361,    0.18766,    0.91986,    0.16172,    0.42798,    0.42351,    0.09759,    0.57199,    0.88008,    0.91165,    0.01692,    0.15625,    0.83097,    0.05542,    0.02426,    0.10352,    0.68465,    0.66088,    0.05391,    0.57460,    0.82047,    0.77920,    0.30227,    0.93930,    0.12136,    0.61019,    0.17454,    0.18187,    0.59105,    0.86176,    0.04962,    0.76691,    0.00390,    0.12911,    0.94865,    0.13755,    0.46775,    0.36593,    0.03971,    0.24682,    0.78608,    0.42105,    0.14576,    0.12255,    0.79879,    0.36818,    0.26877,    0.15829,    0.72135,    0.74132,    0.42383,    0.30345,    0.27988,    0.64650,    0.37735,    0.88838,    0.38920,    0.06370,    0.83166,    0.05019,    0.72419,    0.02747,    0.41240,    0.98674,    0.31811,    0.22564,    0.86001,    0.22105,    0.96910,    0.57640,    0.16101,    0.23709,    0.93247,    0.26274,    0.15488,    0.57962,    0.68815,    0.61049,    0.07429,    0.35141,    0.02402,    0.31978,    0.20938,    0.84776,    0.97740,    0.78252,    0.24890,    0.83358,    0.60182,    0.61401,    0.63872,    0.45299,    0.60102,    0.96819,    0.19311,    0.97186,    0.77109,    0.07006,    0.69307,    0.97076,    0.20988,    0.88694,    0.60953,    0.66253,    0.06539,    0.84395,    0.10136,    0.85947,    0.12390,    0.25344,    0.91984,    0.83278,    0.07913,    0.34430,    0.67687,    0.12911,    0.38563,    0.71542,    0.81710,    0.20750,    0.87524,    0.69305,    0.47794,    0.58671,    0.59402,    0.99788,    0.28932,    0.25082,    0.97629,    0.38778,    0.96444,    0.50776,    0.75407,    0.34873,    0.28818,    0.69716,    0.87002,    0.89752,    0.28954,    0.31314,    0.93149,    0.83766,    0.54063,    0.67452,    0.98125,    0.33621,    0.72906,    0.52275,    0.33221,    0.81217,    0.14737,    0.92647,    0.06536,    0.75226,    0.61785,    0.16375,    0.05600,    0.98444,    0.33410,    0.00962,    0.10776,    0.22222,    0.87150,    0.30929,    0.82402,    0.46748,    0.24905,    0.67018,    0.92796,    0.97353,    0.89513,    0.99308,    0.50498,    0.37357,    0.62527,    0.64767,    0.14303,    0.95402,    0.07071,    0.51975,    0.88764,    0.42337,    0.00338,    0.19116,    0.80465,    0.06187,    0.61640,    0.84480,    0.58404,    0.18081,    0.23644,    0.61126,    0.95712,    0.58446,    0.05959,    0.78148,    0.72375,    0.10536,    0.43701,    0.21053,    0.59954,    0.96552,    0.05924,    0.29656,    0.84916,    0.83230,    0.22892,    0.05503,    0.42318,    0.88057,    0.26604,    0.08475,    0.69183,    0.06894,    0.97888,    0.82705,    0.55562,    0.16311,    0.61890,    0.87547,    0.51232,    0.36556,    0.43792,    0.34088,    0.23209,    0.41270,    0.27762,    0.06855,    0.62692,    0.16242,    0.38135,    0.22200,    0.45952,    0.57922,    0.26642,    0.17071,    0.60652,    0.00784,    0.36553,    0.52969,    0.74624,    0.42126,    0.76395,    0.16900,    0.81700,    0.58921,    0.94714,    0.01487,    0.69578,    0.55007,    0.19971,    0.15534,    0.85581,    0.23413,    0.54158,    0.43867,    0.53179,    0.18648,    0.48030,    0.65265,    0.81593,    0.20711,    0.19905,    0.93623,    0.59942,    0.77110,    0.31425,    0.17555,    0.06693,    0.50739,    0.25808,    0.57057,    0.75263,    0.22761,    0.36659,    0.86707,    0.01701,    0.96888,    0.43597,    0.44806,    0.83158,    0.72255,    0.91108,    0.37273,    0.63119,    0.25343,    0.60080,    0.92568,    0.92272,    0.83908,    0.75614,    0.55297,    0.61713,    0.93788,    0.95334,    0.35952,    0.31443,    0.58770,    0.10495,    0.13540,    0.93460,    0.83917,    0.82064,    0.64114,    0.01350,    0.95963,    0.48750,    0.43861,    0.04079,    0.13069,    0.83948,    0.12936,    0.80176,    0.96411,    0.71667,    0.41876,    0.60577,    0.25623,    0.68416,    0.39622,    0.43669,    0.18543,    0.97554,    0.43688,    0.56838,    0.06289,    0.90969,    0.08574,    0.55542,    0.99918,    0.29854,    0.09029,    0.50880,    0.89964,    0.06607,    0.67432,    0.00311,    0.02116,    0.79777,    0.88196,    0.73259,    0.32523,    0.67480,    0.92425,    0.45190,    0.89836,    0.09585,    0.23926,    0.36707,    0.97505,    0.16401,    0.99041,    0.74082,    0.44658,    0.71707,    0.54924,    0.39856,    0.98721,    0.47258,    0.53250,    0.28350,    0.01008,    0.58560,    0.28032,    0.48058,    0.79544,    0.52324,    0.80748,    0.12537,    0.14986,    0.76543,    0.19069,    0.77942,    0.73050,    0.95193,    0.06030,    0.47038,    0.97578,    0.37574,    0.42857,    0.25427,    0.74734,    0.56612,    0.40995,    0.48026,    0.68474,    0.70917,    0.51616,    0.82167,    0.50286,    0.31501,    0.94874,    0.64787,    0.61927,    0.57627,    0.08965,    0.91441,    0.44538,    0.07515,    0.56602,    0.73711,    0.03229,    0.05112,    0.39480,    0.43184,    0.04176,    0.25867,    0.80482,    0.29571,    0.85680,    0.87032,    0.90377,    0.01783,    0.49946,    0.30916,    0.23609,    0.21791,    0.56503,    0.28941,    0.62817,    0.79302,    0.42292,    0.94860,    0.18815,    0.54023,    0.84242,    0.12998,    0.96324,    0.39855,    0.56021,    0.46704,    0.93231,    0.00374,    0.85542,    0.57586,    0.08907,   -0.71236,   -0.17979,   -0.40240,    0.52685,    0.85554,    0.35431,    0.64717,    0.10409,    0.51494,    0.27458,    0.99131,    0.20203,    0.20510,    0.94026,    0.03760,    0.30233,    0.74509,    0.64520,    0.30108,    0.40339,    0.33822,    0.90746,    0.92682,    0.77360,    0.82459,    0.11637,    0.51181,    0.77742,    0.16776,    0.95711,    0.99800,    0.41171,    0.99882,    0.90537,    0.13494,    0.53352,    0.70355,    0.62640,    0.26277,    0.83004,    0.47465,    0.88184,    0.27077,    0.43572,    0.49693,    0.63458,    0.29639,    0.25042,    0.47479,    0.47311,    0.42073,   -0.66484,   -3.39761,   -0.07964,    0.57491,    0.56582,    0.50659,    0.59887,    0.73491,    0.21909,    0.80989,    0.74626,    0.29166,    0.69472,    0.76110,    0.47229,    0.91144,    0.82928,    0.28991,    0.61384,    0.80752,    0.78863,    0.04124,    0.31743,    0.87438,    0.31514,    0.00083,    0.46612,    0.61837,    0.51295,    0.26267,    0.81355,    0.42422,    0.77015,    0.43578,    0.44947,    0.21051,    0.90181,    0.98115,    0.93611,    0.99665,    0.71164,    0.63390,    0.39688,    0.00404,    0.79939,    0.50197,    0.38885,    0.92556,    0.65674,    0.92075,    0.41218,   -0.66430,    0.46337,   -0.06001,   -0.00423,   -0.00993,    0.59907,    0.84091,    0.41324,    0.40649,    0.11615,    0.67104,    0.51229,    0.92368,    0.27835,    0.28194,    0.07492,    0.25221,    0.43130,    0.42084,    0.27972,    0.80485,    0.62398,    0.92182,    0.61191,    0.78221,    0.28804,    0.78232,    0.68240,    0.43545,    0.76033,    0.40238,    0.47380,    0.58199,    0.34553,    0.44454,    0.15105,    0.63538,    0.32117,    0.31818,    0.99308,    0.58597,    0.39020,    0.89738,    0.65055,    0.72743,    0.08171,    0.21633,    0.34466,    0.66314,    0.42549,    0.34792,    0.02623,    0.37639,    0.08511,   -1.65522,    0.09519,    0.59638,    0.66550,    0.49258,    0.69222,    0.47194,    0.72650,    0.08950,    0.24371,    0.48321,    0.93267,    0.95806,    0.79659,    0.22845,    0.78612,    0.86758,    0.64593,    0.43436,    0.76797,    0.00868,    0.48185,    0.22927,    0.23779,    0.88276,    0.17970,    0.26110,    0.49037,    0.37546,    0.93290,    0.14170,    0.37617,    0.91521,    0.90862,    0.34371,    0.84968,    0.38579,    0.63785,    0.22625,    0.71196,    0.94322,    0.16702,    0.75148,    0.01049,    0.04394,    0.53184,    0.60906,    0.29176,   -0.07108,    0.46205,    0.05807,    0.21650,    0.08665,    0.19582,    0.84764,    0.58251,    0.57805,    0.70657,    0.83488,    0.60219,    0.90700,    0.99881,    0.44620,    0.87053,    0.68681,    0.53921,    0.35231,    0.77352,    0.26924,    0.52793,    0.11250,    0.14125,    0.72380,    0.37154,    0.02174,    0.87693,    0.42913,    0.19017,    0.49551,    0.39718,    0.59385,    0.80029,    0.51058,    0.50937,    0.35578,    0.23999,    0.11160,    0.44118,    0.69209,    0.09051,    0.47264,    0.99397,    0.79662,    0.83367,    0.17110,    0.71865,    0.62089,    0.15544,    0.82803,    0.21334,    0.47167,    0.17192,    0.21791,    0.55530,    0.25260,    0.36095,    0.51998,    0.37958,    0.01123,    0.87958,    0.54308,    0.05393,    0.39522,    0.75107,    0.40162,    0.92498,    0.52519,    0.02399,    0.58229,    0.32251,    0.77264,    0.76160,    0.07715,    0.30037,    0.88873,    0.87691,    0.48225,    0.65200,    0.89344,    0.61074,    0.91862,    0.99860,    0.36900,    0.03926,    0.28597,    0.94480,    0.78874,    0.54334,    0.94488,    0.29378,    0.97465,    0.01307,    0.56627,    0.78915,    0.56257,    0.96467,    0.38766,    0.21012,    0.83809,    0.28628,    0.27771,    0.00353,    0.64311,    0.81301,    0.36695,    0.79172,    0.77298,    0.43399,    0.37422,    0.89668,    0.95819,    0.20529,    0.32437,    0.82929,    0.62256,    0.11366,    0.06349,    0.77204,    0.20538,    0.64176,    0.56687,    0.55100,    0.64925,    0.18327,    0.51141,    0.73067,    0.31137,    0.33231,    0.78789,    0.34489,    0.93607,    0.63039,    0.96477,    0.94959,    0.99304,    0.31270,    0.02583,    0.56074,    0.42745,    0.58277,    0.04250,    0.32883,    0.95183,    0.13371,    0.15238,    0.14650,    0.46050,    0.73987,    0.13135,    0.71064,    0.24583,    0.53760,    0.83969,    0.57397,    0.20781,    0.52547,    0.30436,    0.37117,    0.34875,    0.32099,    0.91114,    0.54492,    0.65504,    0.24742,    0.78125,    0.78860,    0.98970,    0.15761,    0.45133,    0.34195,    0.14152,    0.94289,    0.63796,    0.48937,    0.00127,    0.86666,    0.81244,    0.33872,    0.15069,    0.94326,    0.62029,    0.59788,    0.42075,    0.22841,    0.03090,    0.88041,    0.27244,    0.29265,    0.53011,    0.87595,    0.26720,    0.98816,    0.85105,    0.03015,    0.82169,    0.29700,    0.70143,    0.57811,    0.16379,    0.74021,    0.26418,    0.18218,    0.31060,    0.73302,    0.55659,    0.03759,    0.28731,    0.90059,    0.58750,    0.84151,    0.04900,    0.88925,    0.58760,    0.52767,    0.21206,    0.62219,    0.83580,    0.42788,    0.20243,    0.42224,    0.21959,    0.10895,    0.53642,    0.13127,    0.85854,    0.18724,    0.56768,    0.62417,    0.97248,    0.82369,    0.20718,    0.12234,    0.91872,    0.82177,    0.71811,    0.30182,    0.06697,    0.97780,    0.28142,    0.74791,    0.67239,    0.27209,    0.26760,    0.17829,    0.19265,    0.03753,    0.57882,    0.47665,    0.20503,    0.08717,    0.25253,    0.81580,    0.91722,    0.58167,    0.89073,    0.16552,    0.31350,    0.86255,    0.90073,    0.92368,    0.83097,    0.79077,    0.41170,    0.06121,    0.13313,    0.53861,    0.72203,    0.84561,    0.53246,    0.88248,    0.00490,    0.02869,    0.65777,    0.62479,    0.91274,    0.80402,    0.53007,    0.39717,    0.73200,    0.98169,    0.21906,    0.68938,    0.98005,    0.36949,    0.82513,    0.06677,    0.56075,    0.63931,    0.78098,    0.69408,    0.47125,    0.19325,    0.99542,    0.57795,    0.04838,    0.73681,    0.86310,    0.50254,    0.91462,    0.98746,    0.77352,    0.62918,    0.54091,    0.18081,    0.42875,    0.59749,    0.93143,    0.80416,    0.82887,    0.61990,    0.80176,    0.72795,    0.02813,    0.16369,    0.62069,    0.98179,    0.26690,    0.25614,    0.36689,    0.77086,    0.14216,    0.15281,    0.61252,    0.79811,    0.66516,    0.36541,    0.19991,    0.46752,    0.81670,    0.85520,    0.55059,    0.82016,    0.08649,    0.67210,    0.69285,    0.08319,    0.92759,    0.14990,    0.85547,    0.54450,    0.68392,    0.53576,    0.85577,    0.11912,    0.89081,    0.88526,    0.91246,    0.40514,    0.35025,    0.03456,    0.25407,    0.60780,    0.73324,    0.21208,    0.62763,    0.56445,    0.34674,    0.72456,    0.07559,    0.08059,    0.11067,    0.87993,    0.22663,    0.22224,    0.70126,    0.91679,    0.22461,    0.21145,    0.52056,    0.82228,    0.48509,    0.50441,    0.13314,    0.82310,    0.23785,    0.29896,    0.41307,    0.44498,    0.20371,    0.09681,    0.27299,    0.93259,    0.02647,    0.22269,    0.22096,    0.45271,    0.55761,    0.41466,    0.50640,    0.48639,    0.57916,    0.07696,    0.60286,    0.87647,    0.74040,    0.50883,    0.58681,    0.80209,    0.87507,    0.81730,    0.04035,    0.40173,    0.86198,    0.09172,    0.03932,    0.10946,    0.27233,    0.98673,    0.78450,    0.07356,    0.91111,    0.62124,    0.12645,    0.42291,    0.67686,    0.53192,    0.01897,    0.40615,    0.33871,    0.60384,    0.83981,    0.44556,    0.20686,    0.89429,    0.52046,    0.56334,    0.44590,    0.03722,    0.30937,    0.15705,    0.97091,    0.98407,    0.65850,    0.62101,    0.99234,    0.60192,    0.18984,    0.12887,    0.56816,    0.35804,    0.65028,    0.36076,    0.73196,    0.40132,    0.94902,    0.91647,    0.95116,    0.33825,    0.74407,    0.84822,    0.84665,    0.13436,    0.89094,    0.50748,    0.96106,    0.69589,    0.23002,    0.99872,    0.83593,    0.83218,    0.13199,    0.50704,    0.92624,    0.79538,    0.01238,    0.14945,    0.49373,    0.86742,    0.99856,    0.20740,    0.96566,    0.28343,    0.45855,    0.21399,    0.09819,    0.85187,    0.62619,    0.07579,    0.83057,    0.91802,    0.17061,    0.07258,    0.00377,    0.56024,    0.62112,    0.33424,    0.62610,    0.35011,    0.18915,    0.62230,    0.90079,    0.83377,    0.36924,    0.80161,    0.22331,    0.85695,    0.83947,    0.21063,    0.47432,    0.83563,    0.90394,    0.98528,    0.50777,    0.57040,    0.53145,    0.75928,    0.27085,    0.15615,    0.78263,    0.83454,    0.42879,    0.54447,    0.03243,    0.98361,    0.46858,    0.89675,    0.55673,    0.29383,    0.08206,    0.14820,    0.39402,    0.28378,    0.98872,    0.39445,    0.20701,    0.37540,    0.25011,    0.39199,    0.62967,    0.33591,    0.47757,    0.13107,    0.91087,    0.40502,    0.48335,    0.18577,    0.94967,    0.25682,    0.63589,    0.21724,    0.07908,    0.44786,    0.24993,    0.08979,    0.84074,    0.00537,    0.12513,    0.24966,    0.30816,    0.86959,    0.29068,    0.90377,    0.52045,    0.46265,    0.18707,    0.00643,    0.06013,    0.75020,    0.90771,    0.30597,    0.04247,    0.09102,    0.49229,    0.40213,    0.98014,    0.57864,    0.77576,    0.92495,    0.85192,    0.52930,    0.41201,    0.76800,    0.68827,    0.25020,    0.63748,    0.33538,    0.87672,    0.45303,    0.75330,    0.95776,    0.50760,    0.59763,    0.37598,    0.25342,    0.13180,    0.34855,    0.89702,    0.10256,    0.02748,    0.57691,    0.05118,    0.72142,    0.12656,    0.13251,    0.12062,    0.13408,    0.55815,    0.03493,    0.24581,    0.01705,    0.47200,    0.04764,    0.85978,    0.07998,    0.73593,    0.66281,    0.58510,    0.05849,    0.04183,    0.13271,    0.54244,    0.78240,    0.80334,    0.49073,    0.82249,    0.77311,    0.49856,    0.47347,    0.07613,    0.40192,    0.55251,    0.28315,    0.91788,    0.84729,    0.75041,    0.05556,    0.48177,    0.04987,    0.15621,    0.45895,    0.14863,    0.39329,    0.74485,    0.88593,    0.89194,    0.45058,    0.72319,    0.90636,    0.37967,    0.84766,    0.24820,    0.26119,    0.79797,    0.69382,    0.01091,    0.74738,    0.89610,    0.64285,    0.47935,    0.98524,    0.02333,    0.08500,    0.30809,    0.02168,    0.45146,    0.65536,    0.80215,    0.32163,    0.35571,    0.54712,    0.69751,    0.56246,    0.90902,    0.05758,    0.71148,    0.32857,    0.77339,    0.89172,    0.02293,    0.78824,    0.24597,    0.65630,    0.39383,    0.05546,    0.21914,    0.67581,    0.45250,    0.28851,    0.46458,    0.76001,    0.25631,    0.56515,    0.54901,    0.44947,    0.63355,    0.28899,    0.51477,    0.33767,    0.70065,    0.00754,    0.44849,    0.56643,    0.94994,    0.04065,    0.29647,    0.39725,    0.12596,    0.39617,    0.52114,    0.52581,    0.78430,    0.90592,    0.89216,    0.78921,    0.60163,    0.30805,    0.35131,    0.03182,    0.05360,    0.48100,    0.59430,    0.44883,    0.07986,    0.50432,    0.40026,    0.62432,    0.86840,    0.39588,    0.96915,    0.91931,    0.84528,    0.72085,    0.29164,    0.33385,    0.83510,    0.59908,    0.51179,    0.38189,    0.16004,    0.47477,    0.55540,    0.46613,    0.71033,    0.65041,    0.07786,    0.65503,    0.83663,    0.55779,    0.71229,    0.24913,    0.58479,    0.47307,    0.41239,    0.21174,    0.91528,    0.74766,    0.30751,    0.74764,    0.98822,    0.88834,    0.57367,    0.05880,    0.42237,    0.54737,    0.83511,    0.43354,    0.48171,    0.49686,    0.40846,    0.96386,    0.75997,    0.92257,    0.70885,    0.51446,    0.81913,    0.35645,    0.68910,    0.81460,    0.08872,    0.28630,    0.38046,    0.66482,    0.56614,    0.00434,    0.33444,    0.31902,    0.05948,    0.07148,    0.78122,    0.99526,    0.86872,    0.38472,    0.25443,    0.04423,    0.58559,    0.52178,    0.85321,    0.82836,    0.33844,    0.14466,    0.29139,    0.54861,    0.46442,    0.45135,    0.52305,    0.73148,    0.50479,    0.19596,    0.34691,    0.21861,    0.17078,    0.01761,    0.21584,    0.20679,    0.96869,    0.18589,    0.53205,    0.76757,    0.85604,    0.41044,    0.72217,    0.97964,    0.46762,    0.03317,    0.50410,    0.93682,    0.97963,    0.30497,    0.87355,    0.50625,    0.51829,    0.92567,    0.92360,    0.13604,    0.53672,    0.49327,    0.90471,    0.24975,    0.19696,    0.07718,    0.93436,    0.68704,    0.35992,    0.75252,    0.90280,    0.82618,    0.38615,    0.10359,    0.96931,    0.78729,    0.57337,    0.69464,    0.81332,    0.08548,    0.54148,    0.55730,    0.19755,    0.67313,    0.36674,    0.59139,    0.99581,    0.28505,    0.91979,    0.69475,    0.20338,    0.70614,    0.32810,    0.66672,    0.33756,    0.77188,    0.15874,    0.95786,    0.03703,    0.33158,    0.44129,    0.42006,    0.32345,    0.59458,    0.57697,    0.69475,    0.97553,    0.32841,    0.54930,    0.58753,    0.34768,    0.13620,    0.88390,    0.74645,    0.81211,    0.56647,    0.24165,    0.69838,    0.31720,    0.15548,    0.24281,    0.72769,    0.01770,    0.10726,    0.11133,    0.84029,    0.27316,    0.60391,    0.61718,    0.84481,    0.14273,    0.71786,    0.09828,    0.68370,    0.57613,    0.11839,    0.79170,    0.92873,    0.03894,    0.27610,    0.03809,    0.91122,    0.45671,    0.50720,    0.76430,    0.27914,    0.24791,    0.07811,    0.47629,    0.90125,    0.02198,    0.65245,    0.30107,    0.10783,    0.18970,    0.83499,    0.52129,    0.98822,    0.75832,    0.93570,    0.94472,    0.64323,    0.06603,    0.93045,    0.62588,    0.54758,    0.83386,    0.99624,    0.73722,    0.75616,    0.77211,    0.42712,    0.52253,    0.04937,    0.76119,    0.74183,    0.02918,    0.20786,    0.97993,    0.46591,    0.55999,    0.65716,    0.30510,    0.53639,    0.84158,    0.43209,    0.90665,    0.36660,    0.41438,    0.70076,    0.66683,    0.67137,    0.12524,    0.90948,    0.56386,    0.95125,    0.36351,    0.40501,    0.33659,    0.58832,    0.77041,    0.83098,    0.09593,    0.32279,    0.15033,    0.29984,    0.71904,    0.05461,    0.71132,    0.75460,    0.08249,    0.55183,    0.43933,    0.97058,    0.26050,    0.31382,    0.25214,    0.12383,    0.84564,    0.57343,    0.61394,    0.99131,    0.63282,    0.23008,    0.44687,    0.58798,    0.61719,    0.37027,    0.75278,    0.72723,    0.04420,    0.62620,    0.62160,    0.62324,    0.43745,    0.53486,    0.66298,    0.25198,    0.34331,    0.38870,    0.51020,    0.98593,    0.20472,    0.70167,    0.68533,    0.29631,    0.36038,    0.73030,    0.10049,    0.05042,    0.65152,    0.59604,    0.71042,    0.35305,    0.92270,    0.69103,    0.39806,    0.31635,    0.72084,    0.35561,    0.39478,    0.30876,    0.43427,    0.70224,    0.89373,    0.74445,    0.76571,    0.02822,    0.36219,    0.78814,    0.75818,    0.71485,    0.41030,    0.41121,    0.76804,    0.87729,    0.58335,    0.85875,    0.42658,    0.22873,    0.22660,    0.63549,    0.51682,    0.07717,    0.14797,    0.41216,    0.97818,    0.70149,    0.09644,    0.07445,    0.11324,    0.11924,    0.37664,    0.74916,    0.07026,    0.96294,    0.20441,    0.17097,    0.33061,    0.81664,    0.34826,    0.86173,    0.51761,    0.13948,    0.70857,    0.95723,    0.52313,    0.74465,    0.05720,    0.21376,    0.28472,    0.14358,    0.90946,    0.82194,    0.56283,    0.10407,    0.08162,    0.26500,    0.10965,    0.81303,    0.66768,    0.87013,    0.28677,    0.72766,    0.89914,    0.76799,    0.41449,    0.05025,    0.33697,    0.47724,    0.53639,    0.06247,    0.92668,    0.44641,    0.59323,    0.87323,    0.75501,    0.92981,    0.32054,    0.12463,    0.66346,    0.58672,    0.91976,    0.07720,    0.31181,    0.06399,    0.92664,    0.02015,    0.62469,    0.77571,    0.89229,    0.26438,    0.97925,    0.76741,    0.84449,    0.97947,    0.79782,    0.30182,    0.13778,    0.78083,    0.03787,    0.07549,    0.77862,    0.54316,    0.30485,    0.07877,    0.12720,    0.51172,    0.14206,    0.96666,    0.82941,    0.11406,    0.67563,    0.50128,    0.48969,    0.40920,    0.30346,    0.42378,    0.22181,    0.14975,    0.73631,    0.96391,    0.73196,    0.44394,    0.76051,    0.03317,    0.45666,    0.86451,    0.05220,    0.29947,    0.31073,    0.12349,    0.39867,    0.51871,    0.73426,    0.40127,    0.67691,    0.57470,    0.06278,    0.59752,    0.56129,    0.40804,    0.03766,    0.20388,    0.37785,    0.75522,    0.58773,    0.68104,    0.66138,    0.12961,    0.36102,    0.14969,    0.37973,    0.47046,    0.95513,    0.89571,    0.16503,    0.76942,    0.58554,    0.37342,    0.66633,    0.90500,    0.92210,    0.46259,    0.19937,    0.07053,    0.18773,    0.62181,    0.59532,    0.19777,    0.96954,    0.68579,    0.75975,    0.44140,    0.77016,    0.58516,    0.80766,    0.10617,    0.88478,    0.23050,    0.49455,    0.10145,    0.20489,    0.10319,    0.37501,    0.49759,    0.13779,    0.83447,    0.48111,    0.74707,    0.99751,    0.60500,    0.11010,    0.74397,    0.33128,    0.79508,    0.68879,    0.75486,    0.00414,    0.47300,    0.92942,    0.14702,    0.10613,    0.44475,    0.59962,    0.56555,    0.65202,    0.78286,    0.00655,    0.86313,    0.29131,    0.08029,    0.71674,    0.44508,    0.96963,    0.90179,    0.74792,    0.30654,    0.85455,    0.15136,    0.26939,    0.65825,    0.61864,    0.19397,    0.00610,    0.16009,    0.54932,    0.83302,    0.81194,    0.00547,    0.22139,    0.93068,    0.10512,    0.76008,    0.18649,    0.33430,    0.65589,    0.40949,    0.11696,    0.43357,    0.18994,    0.31009,    0.86121,    0.24609,    0.21266,    0.51164,    0.08707,    0.51961,    0.99233,    0.10387,    0.16386,    0.10488,    0.04847,    0.46970,    0.86070,    0.63727,    0.02791,    0.29713,    0.14782,    0.28620,    0.50174,    0.53296,    0.17431,    0.27462,    0.25275,    0.83076,    0.21610,    0.41946,    0.98322,    0.46414,    0.30206,    0.14733,    0.34713,    0.84324,    0.16303,    0.73760,    0.28062,    0.71966,    0.81947,    0.86258,    0.69208,    0.03015,    0.60342,    0.44704,    0.39840,    0.37707,    0.46783,    0.44261,    0.53066,    0.49899,    0.81530,    0.11037,    0.69571,    0.51956,    0.40019,    0.38999,    0.10184,    0.75591,    0.66004,    0.99458,    0.29175,    0.72696,    0.87313,    0.17409,    0.41035,    0.14179,    0.62325,    0.44007,    0.81320,    0.46701,    0.59207,    0.49343,    0.65489,    0.75378,    0.72940,    0.38474,    0.51683,    0.54213,    0.75238,    0.71892,    0.18584,    0.52303,    0.69104,    0.92116,    0.63554,    0.04812,    0.69366,    0.14826,    0.59839,    0.87533,    0.68338,    0.62519,    0.13747,    0.43883,    0.12711,    0.36874,    0.72869,    0.73269,    0.97561,    0.16531,    0.62811,    0.77808,    0.94204,    0.88530,    0.07896,    0.66608,    0.71684,    0.82514,    0.89293,    0.06113,    0.31812,    0.12038,    0.64508,    0.52251,    0.93316,    0.81871,    0.06365,    0.16386,    0.79106,    0.31259,    0.15602,    0.41418,    0.26396,    0.68670,    0.31822,    0.15479,    0.64555,    0.81650,    0.46604,    0.28886,    0.04784,    0.08131,    0.75758,    0.07214,    0.43759,    0.97832,    0.08546])
print(len(weights))
Mat = weights.reshape((51, 51), order='F')
print "mat ", Mat
mFlat = Mat.flatten()
norm = mFlat/np.linalg.norm(mFlat)
# print("norm is ", norm)
# norm_thresh = np.zeros(len(weights))
# for i in range(len(norm)):
#   if norm[i] >=0.2:
#     norm_thresh[i] = 1
#   else:
#     norm_thresh[i] = 0.1

# minMaxMat = np.zeros_like(Mat)
# for i, col in enumerate(Mat.T):
#     mx = np.max(col)
#     for j, row in enumerate(col):
#         if Mat[j][i] == mx:
#             minMaxMat[i][j] = 1
#         else:
#             minMaxMat[i][j] = 0
# print minMaxMat.T
# minMaxMat = minMaxMat.T
#
# mFlat = minMaxMat.flatten()
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(weights)):
  trajectories_z0[i][2] = norm[i]
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
fig.savefig("/home/vignesh/Desktop/reward_plot")