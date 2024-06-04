import numpy as np
import torch


def Action_adapter(a, max_action):
    # from [-1,1] to [-max,max]
    return a * max_action


def Action_t_relu6_adapter_reverse(act, min_time, max_action):
    # from [0, max] to [0,6]
    return ((act - min_time) * 6.0) / (max_action - min_time)


def Action_t_relu6_adapter(a, min_time, max_action):
    # from [0,6] to [0,max]
    return (a / 6.0) * (max_action - min_time) + min_time


def Action_t_sigmoid_adapter_reverse(act, min_time, max_action):
    # from [0, max] to [0,6]
    return ((act - min_time) * 1.0) / (max_action - min_time)


def Action_t_sigmoid_adapter(a, min_time, max_action):
    # from [0,6] to [0,max]
    return (a / 1.0) * (max_action - min_time) + min_time


def Action_t_sigmoid_adapter_reverse_ori(act):
    # from [0, max] to [0,6]
    if act == 1.0 / 75.0:
        a = np.random.uniform(0.67, 1.0)
    elif act == 1.0 / 50.0:
        a = np.random.uniform(0.34, 0.67)
    else:
        a = np.random.uniform(0.0, 0.34)
    return np.array([a])


def Action_t_sigmoid_adapter_ori(a):
    # from [0,6] to [0,max]
    if 0.67 <= a <= 1.0:
        act = 1.0 / 75.0
    elif 0.34 <= a < 0.67:
        act = 1.0 / 50.0
    else:
        act = 1.0 / 25.0
    return act


def Action_adapter_reverse(act, max_action):
    # from [-max,max] to [-1,1]
    return act / max_action


def Speed_dirction_setter(act, speed_unit):
    if act[0] < 0 and speed_unit[0] < 0:
        speed_unit_x = speed_unit[0]
    elif act[0] < 0 <= speed_unit[0]:
        speed_unit_x = speed_unit[0] * -1.0
    elif act[0] >= 0 and speed_unit[0] >= 0:
        speed_unit_x = speed_unit[0]
    else:
        speed_unit_x = speed_unit[0] * -1.0

    if act[1] < 0 and speed_unit[1] < 0:
        speed_unit_y = speed_unit[1]
    elif act[1] < 0 <= speed_unit[1]:
        speed_unit_y = speed_unit[1] * -1.0
    elif act[1] >= 0 and speed_unit[1] >= 0:
        speed_unit_y = speed_unit[1]
    else:
        speed_unit_y = speed_unit[1] * -1.0
    return np.array([speed_unit_x, speed_unit_y], dtype='float')


def Act_v_correction(act_v):
    # from[-2,2] to [0,2]
    return abs(act_v)


def Act_t_correction(act_t):
    # from[-1,1] to [0,1]
    return abs(act_t)


def update_value_list(current_list, current_value, max_length_of_list):
    if len(current_list) < max_length_of_list:
        current_list.append(current_value)
    else:
        del (current_list[0])
        current_list.append(current_value)
    return current_list


class QueueUpdater:

    def __init__(self, c=1):
        self.capacity = c
        self.data = [0 for i in range(self.capacity)]
        self.tail = 0
        self.len = 0

    def append(self, e):
        if self.len < self.capacity:
            self.data[self.len] = e
            self.len += 1
        else:
            self.data[self.tail] = e
            self.tail = (self.tail + 1) % self.capacity

    def to_numpy(self):

        t = np.zeros([self.capacity])
        for i in range(self.len):
            t[i] = self.data[(i + self.tail) % self.capacity]
        return t

    def reset(self):
        self.data = [0 for i in range(self.capacity)]
        self.tail = 0
        self.len = 0
