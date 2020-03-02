import numpy as np
from matplotlib import pyplot as plt
from robot_state_utils import RobotStateUtils
from numpy import savetxt

rewards1 = np.array([  0.14466,    0.79204,    0.17226,    0.68862,    0.49568,    0.02016,    0.14331,    0.68657,    0.92048,    0.97575,    0.78240,    0.22957,    0.26371,    0.06770,    7.02270,    4.49635,    0.52471,    0.82279,    0.02338,    0.53533,    0.36364,    0.93615,    0.69857,    0.99727,    0.56678,    4.31316,    0.41392,    0.72895,    0.50636,    0.77596,    0.10292,    0.48624,    0.10435,    0.71725,    0.87887,    0.59717,    0.33071,    0.30118,    0.74643,    0.99683,    0.73547,    0.81157,    0.67712,    0.00255,    0.76167,    0.43444,    0.54675,    0.09521,    0.47681,    0.53266,    0.42340,    0.63103,    0.70824,    0.40331,    0.98317,    0.04605,    0.24003,    0.67020,    0.37629,    0.55876,    0.44984,    0.94561,    0.13272,    0.40364,    0.16894,    0.12727,    0.87002,    0.58231,    0.15229,    0.86235,    0.80894,    0.14873,    0.62786,    0.08917,    0.87566,    0.31251,    0.95497,    0.34521,    0.91892,    0.45065,    0.94827,    0.08275,    0.03920,    0.02983,    0.70836,    0.22617,    0.15576,    0.86830,    0.27732,    0.39575,    0.26824,    0.23099,    0.64050,    0.95418,    0.78036,    0.42483,    0.12131,    0.81504,    0.25451,    0.25474,    0.96569,    0.35586,    0.73075,    0.04820,    0.20054,    0.27325,    0.38128,    0.69461,    0.54186,    0.54285,    0.73923,    0.16634,    0.96671,    0.49634,    0.33756,    0.33630,    0.64878,    0.68960,    0.92841,    0.25794,    0.19396,    0.16482,    0.76860,    0.62315,    0.82575,    0.10421,    0.30103,    0.78049,    0.32382,    0.85955,    0.19713,    0.27961,    0.89482,    0.70822,    0.06749,    7.60896,    3.52938,    0.18301,    0.66137,    0.21469,    0.56247,    0.46562,    0.50331,    0.77232,    0.29098,    0.74320,    3.71411,    0.21125,    0.30353,    0.77913,    0.54259,    0.73282,    0.81421,    0.86360,    0.49504,    0.24610,    4.31485,    4.25909,    0.19749,    0.36774,    0.14836,    0.41279,    0.90314,    0.20501,    0.63808,    0.00883,    0.15620,    0.34872,    0.26425,    0.04128,    0.12657,    0.99620,    0.01724,    0.18568,    0.78237,    0.30562,    0.37563,    0.17988,    0.73711,    0.31248,    0.24220,    0.70789,    0.27256,    0.86265,    0.06595,    0.06628,    0.39702,    0.40070,    0.24599,    0.31383,    0.59548,    0.47392,    0.43727,    0.40862,    0.90660,    0.37878,    0.96416,    0.57497,    0.68426,    0.69720,    0.45619,    0.23109,    0.20354,    0.09480,    0.40076,    0.23941,    0.48810,    0.01826,    0.99737,    0.82334,    0.68772,    0.66717,    0.04362,    0.11806,    0.66821,    0.46248,    0.45508,    0.16122,    0.54584,    0.57866,    0.23395,    0.42738,    0.50342,    0.04828,    0.19385,    0.90995,    0.92111,    0.22095,    0.66525,    0.49300,    0.11745,    0.98545,    0.48478,    0.46458,    0.28846,    0.41687,    0.37102,    0.57849,    0.37143,    0.30907,    0.91708,    0.94068,    0.14684,    0.07739,    0.27337,    0.20435,    0.74087,    0.64676,    0.56217,    0.90741,    0.43852,    0.84768,    0.53977,    0.22546,    0.15910,    0.18768,    4.31104,    0.58017,    0.50141,    0.92569,    0.75226,    0.67091,    0.52344,    0.98046,    0.49767,    0.16573,    0.27194,    0.56099,    0.14591,    0.54383,    0.05149,    0.75331,    0.13778,    0.17507,    0.00550,    0.36525,    0.12408,    4.23509,    0.28038,    0.70117,    0.55507,    0.49290,    0.33276,    0.45192,    0.21118,    0.93513,    0.14665,    0.79850,    3.64056,    0.26573,    0.14698,    0.09905,    0.75398,    0.41111,    0.86815,    0.46828,    0.44754,    0.17113,    0.26946,    0.32933,    0.89386,    0.71156,    0.84457,    0.62072,    0.62759,    0.94350,    0.98119,    0.85641,    0.43883,    0.77340,    0.47154,    0.59751,    0.98695,    0.60484,    0.55688,    0.95720,    0.42647,    0.51759,    0.27845,    0.83828,    0.73221,    0.49162,    0.91277,    0.31182,    0.30882,    0.76784,    0.44907,    0.94461,    0.62012,    0.90822,    0.20694,    0.20751,   -0.02801,    0.19904,    0.69872,    0.19727,    0.60127,    0.63201,    0.59478,    0.78722,    0.44128,    0.79757,    0.85501,    0.81870,    0.56810,    0.38621,    0.96132,    0.53441,    0.36073,    0.37453,    0.92244,    0.02823,    0.23458,    0.72646,   -1.82981,    0.29001,    0.14870,    0.30228,    0.06487,    0.76715,    0.21602,    0.30519,    0.23546,    0.86885,    0.15749,    0.36030,    0.29523,    0.26070,    0.82475,    0.28514,    0.12249,    0.51332,    0.03860,    0.38745,    0.38411,    0.69922,    0.97744,    3.93122,    4.25318,    0.62839,    0.26459,    0.32507,    0.28446,    0.03992,    0.43879,    0.99616,    0.00442,    0.69235,    0.22190,    0.69277,    0.27884,    0.74564,    0.54052,    0.26824,    0.27472,    0.88580,    0.30448,    0.32224,    0.53084,    0.84001,    0.92233,    0.71134,    0.62965,    0.99269,    0.76623,    0.47099,    0.16243,    0.09433,    0.35487,    4.11830,    0.44174,    0.14720,    0.45196,    0.20449,    0.36776,    0.75263,    0.26256,    0.40720,    0.14833,    3.86111,    3.83393,    0.25525,    0.56505,    0.86391,    0.35697,    0.87539,    0.77986,    0.97913,    0.57559,    0.26515,    0.47466,    0.28738,    0.98809,    0.19975,    0.19117,    0.31432,    0.61146,    0.56268,    0.85014,    0.10258,    0.19216,    0.31620,    0.61561,    0.47724,    0.43441,    0.25419,    0.24542,    0.53632,    0.25413,    0.38561,    0.96332,    0.12645,   -1.84345,   -0.08311,    0.47596,    0.15401,    0.60713,    0.30811,    0.59139,    0.38031,    0.64191,    0.88764,    0.29210,    0.07111,    0.20216,    0.50969,    0.22371,    0.85018,    0.28295,    0.44592,    0.25617,    0.66749,    0.92806,    0.86221,    0.04465,    0.00317,    0.33521,    0.70766,    0.52945,    0.35791,    0.09280,    0.55756,    0.32618,    0.65636,    0.26403,    0.87997,    0.40855,    0.20958,    0.19062,    0.38989,    0.82791,    0.04680,    0.00901,    0.33122,    0.93634,    0.48302,    0.11153,    0.15918,    0.71575,    4.42562,    0.04280,    0.12968,    0.23105,    0.64175,    0.97606,    0.84640,    0.46458,    0.98643,    0.40663,    0.96908,    4.19223,    0.55191,    0.63934,    0.59962,    0.08982,    0.50487,    0.58814,    0.55515,    0.09448,    0.25931,    0.23058,    0.55928,    0.29787,    0.43208,    0.47262,    0.47042,    0.02718,    0.65703,    0.38720,    0.82050,    0.04635,    0.62920,    0.34289,    0.81455,    0.97971,    0.13633,    0.46039,    0.28678,    0.74469,    0.57144,    0.71446,    0.70585,    0.17425,    0.92935,    0.91389,    0.80831,    0.14782,    0.62315,    0.51954,    0.51876,    0.13324,    0.21010,    0.89627,    0.17232,    0.66784,    0.42009,    0.85101,    0.14473,    0.75516,    0.00705,    0.49431,    0.76369,    0.49570,    0.33076,    0.26452,    0.13285,    0.47652,    0.45738,    0.89462,    0.84403,    0.27746,    0.97731,    0.33468,    0.21436,    0.07865,    0.99931,    0.89802,    0.02644,    0.66930,    0.35161,    0.22730,    0.68875,    0.32448,    0.60714,    0.84272,    0.66412,    0.16302,    0.89427,    0.05471,    0.71054,    0.76416,    0.23263,    0.49471,    0.53980,   -1.84289,    0.02821,    0.01560,    0.48021,    0.51006,    0.83818,    0.41149,    0.88057,    0.78418,    0.12549,    0.57350,    0.36536,    0.37567,    0.30755,    0.69377,    0.26089,    0.47466,    0.93841,    0.95147,    0.73020,    0.32690,    0.08798,    0.16459,    0.63007,    0.96262,    0.25935,    0.41126,    0.12741,    0.53256,    0.55186,    0.22860,    0.15832,    0.91221,    0.14599,    0.26447,    0.70525,    0.26576,    3.86432,    0.76464,    0.50984,    0.19439,    0.15753,    0.54493,    0.26171,    0.26654,    0.70142,    0.98752,    0.21569,    4.32787,    0.31211,    0.51930,    0.10137,    0.22268,    0.72026,    0.67641,    0.47860,    0.53345,    0.09533,    0.82664,    4.40204,    0.72704,    0.35070,    0.85016,    0.04296,    0.21606,    0.44149,    0.72646,    0.50002,    0.40236,    0.76224,    0.52788,    0.84666,    0.18363,    0.70697,    0.30168,    0.99782,    0.50569,    0.25090,    0.74531,    0.62062,    0.49160,    0.57851,    0.81889,    0.25999,    0.19253,    0.26199,    0.45876,    0.71772,    0.45478,    0.44524,    0.67501,    0.83180,    0.03839,    0.92773,    0.97167,    0.39922,    0.86294,    0.07793,    0.55333,    0.29308,    0.34616,    0.84410,    0.18633,    0.88044,    0.38176,    0.90683,    0.78750,    0.16431,    0.35112,    0.63247,    0.58553,    0.47784,    0.05741,    0.09024,    0.13068,    0.22957,    0.17601,    0.64640,    0.93976,    0.89358,    0.89088,    0.54953,    0.75567,    0.82501,    0.70250,    0.25750,    0.64767,    0.72402,    0.15309,    0.64943,    0.63714,    0.92854,    0.20223,    0.21037,    0.36824,    0.71000,    0.98653,    0.75018,    0.45729,    0.92591,    0.49212,    0.75387,    0.42890,    0.26260,    0.56294,    0.38276,    0.69344,    0.83839,    0.30542,    0.13831,    0.16769,    0.26284,    0.83855,    0.52760,    0.58001,    0.58928,    0.30892,    0.32667,    0.22925,    0.83901,    0.75977,    0.17316,    0.03016,    0.10721,    0.79173,    0.48074,    0.68019,    0.25352,    0.97140,    0.50124,    0.76528,    0.84643,    0.17577,    0.89991,    0.16306,    0.86145,    0.30013,    0.12504,    0.36442,    0.23458,    2.08806,    0.66595,    0.56553,    0.64854,    0.20051,    0.38422,    0.41390,    0.96326,    0.20789,    0.03627,    3.58228,    0.63236,    0.97283,    0.71213,    0.77103,    0.65143,    0.06758,    0.89985,    0.38008,    0.35212,    0.54577,    0.36651,    0.12475,    0.31996,    0.13395,    0.83705,    0.12586,    0.90464,    0.84137,    0.53539,    0.32950,    0.24361,    0.77372,    0.94491,    0.22778,    0.61214,    0.94492,    0.67119,    0.44379,    0.37609,    0.66707,    0.92768,    0.59634,    0.99342,    0.61080,    0.58994,    0.46915,    0.92874,    0.34342,    0.39505,    0.54522,    0.57769,    0.16768,    0.15460,    0.00239,    0.79140,    0.52982,    0.11988,    0.06973,    0.78011,    0.83331,    0.99095,    0.92528,    0.04527,    0.65092,    0.47509,    0.42419,    0.72200,    0.89900,    0.62043,    0.53478,    0.10489,    0.91843,    0.81891,    0.36257,    0.62362,    0.41691,    0.64025,    0.04378,    0.39334,    0.49853,    0.82623,    0.62925,    0.05313,    0.76192,    0.54933,    0.39665,    0.49360,    0.00508,    0.84342,    0.53216,    0.05794,    0.55820,    0.12919,    0.28607,    0.23301,    0.34458,    0.41086,    0.25324,    0.86425,    0.93184,    0.62082,    0.04587,    0.12467,    0.34709,    0.06010,    0.36505,    0.92214,    0.97831,    0.12723,    0.11835,    0.87185,    0.86437,    0.26350,    0.51834,    0.09598,    0.76760,    0.05669,    0.67526,    0.14097,    0.48346,    3.90676,    0.55294,    0.23507,    0.37672,    0.26943,    0.09405,    0.47932,    0.41149,    0.61703,    0.07851,    3.79487,    3.92117,    0.22305,    0.01820,    0.42107,    0.21102,    0.61776,    0.67051,    0.69322,    0.58807,    0.27204,    0.76583,    0.97032,    0.13300,    0.96200,    0.03954,    0.22301,    0.16547,    0.52099,    0.79892,    0.34493,    0.28135,    0.79317,    0.04126,    0.39120,    0.82041,    0.71413,    0.40802,    0.88355,    0.83772,    0.19735,    0.77529,    0.80149,    0.15993,    0.88721,    0.96496,    0.25291,    0.48890,    0.27397,    0.23182,    0.28901,    0.29636,    0.13390,    0.90897,    0.83998,    0.53372,    0.24771,    0.89830,    0.60965,    0.66211,    0.59493,    0.88760,    0.19891,    0.31377,    0.72387,    0.14361,    0.37647,    0.90283,    0.67705,    0.11160,    0.00095,    0.07559,    0.70366,    0.98341,    0.83506,    0.47589,    0.21945,    0.63337,    0.81437,    0.57157,    0.72929,    0.07962,    0.94071,    0.49343,    0.74488,    0.88774,    0.94345,    0.66210,    0.79226,    0.05703,    0.92284,    0.25067,    0.84569,    0.60186,    0.78722,    0.79697,    0.32886,    0.38599,    0.60688,    0.09804,    0.22945,    0.93913,    0.64295,    0.82959,    0.38731,    0.12059,    0.25750,    0.64287,    0.83800,    0.50934,    0.84123,    0.45046,    0.68797,    0.57526,    0.87532,    0.86211,    0.43686,    0.53780,    0.53880,    0.99877,    0.67805,    0.81041,    0.67424,    0.26856,    0.23039,    0.99856,    0.35965,    0.76274,    0.97454,    0.83132,    0.80830,    0.34923,    0.86887,    0.46343,    0.80142,    0.28831,    0.02871,    0.24306,    0.53595,    0.51492,    0.16204,    0.42743,    0.98766,    0.07473,    0.24855,    0.05589,    0.21739,    0.07049,    0.39387,    0.69736,    0.30787,    0.26286,    0.12057,    0.03412,    0.43189,    0.11399,    0.62799,    0.56370,    0.78545,    0.42285,    0.72645,    0.79994,    0.41129,    0.72798,    0.99622,    0.62017,    0.99710,    0.53278,    0.89151,    0.99314,    0.45905,    0.90643,    0.53777,    0.23322,    0.26558,    0.92710,    0.95001,    0.87066,    0.02294,    0.05871,    0.64434,    0.31137,    0.19768,    0.63801,    0.43504,    0.29398,    0.26880,    0.81220,    0.09693,    0.95107,    0.92364,    0.05962,    0.25610,    0.15709,    0.89779,    0.35155,    0.17997,    0.92090,    0.89919,    0.54770,    0.91812,    0.48049,    0.17607,    0.25483,    0.55081,    0.34977,    0.30289,    0.69851,    0.68050,    0.77420,    0.36451,    0.69700,    0.98954,    0.85412,    0.03272,    0.02647,    0.74361,    0.60863,    0.31636,    0.08525,    0.59731,    0.58127,    0.94855,    0.30498,    0.81557,    0.16736,    0.07865,    0.36373,    0.65558,    0.84418,    0.62333,    0.55404,    0.71983,    0.10041,    0.46576,    0.70732,    0.58272,    0.79464,    0.73491,    0.73614,    0.94137,    0.49473,    0.86843,    0.65367,    0.34440,    0.27474,    0.02932,    0.97583,    0.70098,    0.76786,    0.76126,    0.93235,    0.88520,    0.14095,    0.77311,    0.18248,    0.02704,    0.60316,    0.36483,    0.05174,    0.97270,    0.92909,    0.35809,    0.17127,    0.37985,    0.97571,    0.22884,    0.09039,    0.34906,    0.17807,    0.16318,    0.48943,    0.83979,    0.77730,    0.60839,    0.40402,    0.57637,    0.64524,    0.63817,    0.73298,    0.76494,    0.59670,    0.25393,    0.32770,    0.51558,    0.57417,    0.62088,    0.68283,    0.85778,    0.33466,    0.18861,    0.60520,    0.51276,    0.95154,    0.43804,    0.38523,    0.32664,    0.69752,    0.72274,    0.04838,    0.33001,    0.69507,    0.62530,    0.08890,    0.28061,    0.82547,    0.71455,    0.09764,    0.32946,    0.61950,    0.34411,    0.86800,    0.53266,    0.25519,    0.23744,    0.44460,    0.72243,    0.07897,    0.23140,    0.41814,    0.69462,    0.73948,    0.54581,    0.96396,    0.89242,    0.77342,    0.95417,    0.88585,    0.18073,    0.64904,    0.24905,    0.19607,    0.69640,    0.72050,    0.18958,    0.77117,    0.09738,    0.17357,    0.26467,    0.10735,    0.97909,    0.94390,    0.89372,    0.45754,    0.47206,    0.83702,    0.39035,    0.92120,    0.16318,    0.21386,    0.35001,    0.18001,    0.92612,    0.13604,    0.36340,    0.53532,    0.44661,    0.96535,    0.38862,    0.69335,    0.67421,    0.21270,    0.84499,    0.38136,    0.01473,    0.93853,    0.51066,    0.11743,    0.40908,    0.48748,    0.26038,    0.04317,    0.02618,    0.18355,    0.46705,    0.28558,    0.85912,    0.45939,    0.66773,    0.28785,    0.72397,    0.35608,    0.76568,    0.38931,    0.48095,    0.16494,    0.87158,    0.36096,    0.24685,    0.89146,    0.44754,    0.14838,    0.99230,    0.67987,    0.34287,    0.35377,    0.98827,    0.36244,    0.68371,    0.46086,    0.76060,    0.17935,    0.36298,    0.72136,    0.73168,    0.74184,    0.70678,    0.84225,    0.50922,    0.89182,    0.56543,    0.67511,    0.09908,    0.82759,    0.27038,    0.80484,    0.85865,    0.58339,    0.90185,    0.73641,    0.27611,    0.18892,    0.80706,    0.17905,    0.50104,    0.96208,    0.56731,    0.55609,    0.36820,    0.92601,    0.96338,    0.95085,    0.83874,    0.00705,    0.90522,    0.76521])


if __name__ == '__main__':
    filename = "/home/vignesh/Desktop/individual_trials/version2/data1/policy_grid11.txt"
    # term_state = np.random.randint(0, grid_size ** 3)]
    goal = np.array([0.005, 0.055, -0.125])
    env_obj = RobotStateUtils(11, 0.8, goal)
    states = env_obj.create_state_space_model_func()
    action = env_obj.create_action_set_func()
    print "State space created is ", states
    print "actions is ", action
    index = env_obj.get_state_val_index(goal)
    print index, states[index-1], states[index], states[index+1]
    # map magic squares to their connecting square
    P_a = env_obj.get_transition_mat_deterministic()
    policy = env_obj.value_iteration(rewards1)
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