import rosbag
import rospy
import numpy as np
import csv
import math
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def getData(bag, topic):
    """
    Extract a pos from the a topic
    :param bag: path to bag file
    :param topic: name of topic
    :return: the pos from that topic in a Nx3 array
    """
    cur_bag = rosbag.Bag(bag)

    t_start = rospy.Time(cur_bag.get_start_time())
    t_end = rospy.Time(cur_bag.get_end_time())

    num_msgs = cur_bag.get_message_count(topic)
    data = np.zeros(shape=(num_msgs, 3))
    idx = 0

    for top, msg, t in cur_bag.read_messages(topics=[topic],
                                             start_time=t_start,
                                             end_time=t_end):
        cur = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        data[idx, :] = cur
        idx = idx + 1

    return data



def getData(bag, topic, pos=1):
    """
    Extract a pos from the a topic
    :param bag: path to bag file
    :param topic: name of topic
    :return: the pos from that topic in a Nx3 array
    """
    cur_bag = rosbag.Bag(bag)

    t_start = rospy.Time(cur_bag.get_start_time())
    t_end = rospy.Time(cur_bag.get_end_time())

    num_msgs = cur_bag.get_message_count(topic)
    if pos:
        data = np.zeros(shape=(num_msgs, 3))
    else:
        data = np.zeros(shape=(num_msgs, 4))
    idx = 0

    for top, msg, t in cur_bag.read_messages(topics=[topic],
                                             start_time=t_start,
                                             end_time=t_end):
        if pos:
            cur = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        else:
            cur = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w ])
        data[idx, :] = cur
        idx = idx + 1

    return data


def make_steps(data):
    """
    Finds the steps between the data points
    :param data: numpy array
    :return: array steps between the data points
    """
    steps = []
    fixed = []
    curr = data[0]
    count = 0
    steps.append(curr-curr)
    fixed.append(np.round(curr,3))
    for ii in xrange(0, len(data)-1):
        curr = np.round(data[ii],3)
        nxt = np.round(data[ii+1],3)
        # curr = round_nearest(data[ii], 0.005)
        # nxt = round_nearest(data[ii+1], 0.005)
        interval = np.round(nxt-curr, 3)
        if abs(interval[0]) > 0 or abs(interval[1]) > 0 or abs(interval[2]) > 0:
            steps.append(interval)
            fixed.append(curr)

    return np.array(fixed), steps


def make_file(data, steps, name):
    """
    Writes the data into a CSV file
    :param data: data nx3
    :param steps: steps betwen data Nx3
    :param name: name of CSV file
    :return: None
    """
    with open(name + '.csv', mode='w') as f:
        writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['X', 'X_step', 'Y', 'Y_step', 'Z', 'Z_step' ])

        for d, step in zip(data, steps):
            writer.writerow([ d[0], step[0], d[1], step[1], d[2], step[2]])

def round_nearest(x, a):
    return np.round(np.round(x / a) * a, -int(math.floor(math.log10(a))))



if __name__ == "__main__":

    file = "/home/vignesh/Thesis_Suture_data/trial2/suture_data_trial2/2020-02-25_20:02:45.832953.bag"
    topic = "/dvrk/PSM2/position_cartesian_current"
    file_name = "/home/vignesh/Thesis_Suture_data/trial2/suture_data_trial2/781266"
    data = getData(file, topic)
    rnd, steps = make_steps(data)
    #make_file(rnd, steps, file_name)
    plt.rcParams.update({'font.size': 11})
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_title("Discretized data of suturing")
    ax.set_xlabel('X(mm)', labelpad=5)
    ax.set_ylabel('Y(mm)', labelpad=5)
    ax.set_zlabel('Z(mm)', labelpad=5)
    plt.xticks([-30, -20, -10, 0, 10])
    plt.yticks([35, 40, 45, 50, 55])
    ax.set_zticks([-120, -124, -128, -132])
    # plt.xticks([-0.03, -0.02, -0.01, 0, 0.010])
    # plt.yticks([0.035, 0.04, 0.045, 0.05, 0.055])
    # plt.yticks([0.005, 0.015, 0.025, 0.035, 0.045])
    # ax.set_zticks([-0.12, -0.124, -0.128, -0.132, -0.136])
    color_rl = 'b'
    color_user = 'tab:orange'
    ax.plot(rnd[:, 0]*1000, rnd[:, 1]*1000, rnd[:, 2]*1000, color=color_rl, linewidth=2)
    ax.plot(data[:, 0]*1000, data[:, 1]*1000, data[:, 2]*1000, color=color_user, linewidth=2)
    ax.legend(["Discretized", "Raw"], loc=0, bbox_to_anchor=(0.5, 0.5, 0.5, 0.5))
    plt.savefig('/home/vignesh/PycharmProjects/motion_planning_max_entropy_irl/RL_3d_gridsize_11/different_policies/raw_discrete.png')
    plt.show()

