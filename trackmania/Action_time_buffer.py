# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 16:38:13 2023

@author: Dong
"""
import numpy as np

class ActionTimeBuffer(object):

    def __init__(self, c=3):
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

        