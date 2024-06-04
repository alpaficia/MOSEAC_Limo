from SEAC import SEACAgent
from ReplayBuffer import RandomBuffer
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from ReplayBuffer import device
from gymnasium.envs.registration import register
import numpy as np
import gymnasium
import torch
import os
import shutil
import argparse
import time
print(device)

class LimoTrainer(object):
    def __init__(self):
        self.a_lr = 3e-5
        self.c_lr = 3e-5
        self.gamma = 0.99
        self.write = True
        self.save_interval = 1e4
        self.eval_interval = 1e4
        self.eval_turn = 5
        self.update_every = 50
        self.net_width = 256
        self.batch_size = 256
        self.alpha = 0.12
        self.adaptive_alpha = True
        self.fixed_freq = False
        self.obs_freq = 20.0
        self.ep_max_length = 1000
        self.state_dim = 42
        if self.fixed_freq:
            self.action_dim = 2
        else:
            self.action_dim = 3
        kwargs = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "hid_shape": (self.net_width, self.net_width),
            "a_lr": self.a_lr,
            "c_lr": self.c_lr,
            "batch_size": self.batch_size,
            "alpha": self.alpha,
            "adaptive_alpha": self.adaptive_alpha
        }
        self.model = SEACAgent(**kwargs)
