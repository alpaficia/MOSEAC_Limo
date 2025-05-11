import numpy as np
import math


def Action_adapter(a, max_action):
    # from [-1,1] to [-max,max]
    return a * max_action


def Action_t_relu6_adapter_reverse(act, max_action):
    # from [0, max] to [0,6]
    return (act * 6.0) / max_action


def Action_t_relu6_adapter(a, max_action):
    # from [0,6] to [0,max]
    return (a/6.0) * max_action


def Action_adapter_reverse(act, max_action):
    # from [-max,max] to [-1,1]
    return act / max_action


def reward_adapter(reward, alpha_t, alpha_epsilon, min_time, time):
    reward_time = min_time / time
    return float(alpha_t * reward * reward_time - alpha_epsilon)

def sigmoid(x, k, x0):
    return 1 / (1 + math.exp(-k * (x - x0)))

def update_gain_r(alpha_m, increase_amount):
    # 更新 self.gain
    alpha_m = alpha_m + increase_amount
    # 使用逆向 Sigmoid 函数更新 self.gain_eps
    # 选择k和x0的值以确保在gain为1时gain_eps为0.1
    k = 1  # 控制sigmoid函数的斜率
    x0 = 1  # 控制sigmoid中心为1，使得gain为1时输出特定值
    # 更新self.gain_eps，将sigmoid输出乘以10，并减去1使得gain为1时gain_eps为0.1
    alpha_eps = 0.02 * (1 - sigmoid(alpha_m, k, x0))
    return float(alpha_m), float(alpha_eps)

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
        
