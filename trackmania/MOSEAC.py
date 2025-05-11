import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from math import floor
from ReplayBuffer import device
import os

save_path = '/model/'
load_path = '/model/'
dir_path = os.getcwd()
save_path = dir_path + save_path
load_path = dir_path + load_path


def build_net(layer_shape, activation, output_activation):
    # Build net with for loop
    layers = []
    for j in range(len(layer_shape) - 1):
        if j < len(layer_shape) - 2:
            act = activation
        else:
            act = output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j + 1]), act()]
    return nn.Sequential(*layers)

def num_flat_features(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out


class VanillaCNN_Actor(nn.Module):
    def __init__(self, state_dim_without_imgs, img_shape, action_dim, hid_shape, h_acti=nn.ReLU, o_acti=nn.ReLU):
        super(VanillaCNN_Actor, self).__init__()
        self.action_dim = action_dim
        self.h_out, self.w_out = img_shape[0], img_shape[1]
        self.len_of_history_images = 4
        self.conv1 = nn.Conv2d(self.len_of_history_images, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels
        self.flat_features = self.out_channels * self.h_out * self.w_out
        layers = [state_dim_without_imgs + self.flat_features] + list(hid_shape)
        self.a_net = build_net(layers, h_acti, o_acti)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state, imgs, deterministic=False, with_logprob=True):
        # Network with Enforcing Action Bounds
        x = F.relu(self.conv1(imgs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        x = torch.cat((state, x), -1)
        net_out = self.a_net(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        if deterministic:
            u = mu
        else:
            u = dist.rsample()  # re-parameterization trick of Gauss
        a_t = F.relu6(u[:, 0]).unsqueeze(1)
        a_m = torch.tanh(u[:, 1:])
        a = torch.cat([a_t, a_m], dim=1)

        if with_logprob:
            # get probability density of logp_pi_a from probability density of u, which is given by the SAC V2 paper.
            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(
                axis=1, keepdim=True)
        else:
            logp_pi_a = None
        return a, logp_pi_a


class VanillaCNN_QCritic(nn.Module):
    def __init__(self, state_dim_without_imgs, img_shape, action_dim, hid_shape):
        super(VanillaCNN_QCritic, self).__init__()
        self.action_dim = action_dim
        self.h_out, self.w_out = img_shape[0], img_shape[1]
        self.len_of_history_images = 4
        self.conv1 = nn.Conv2d(self.len_of_history_images, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels
        self.flat_features = self.out_channels * self.h_out * self.w_out
        layers = [state_dim_without_imgs + self.flat_features + action_dim] + list(hid_shape) + [1]
        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, imgs, action):
        x = F.relu(self.conv1(imgs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.squeeze(-1)
        x = x.squeeze(-1)
        sa = torch.cat((state, x, action), -1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)
        return q1, q2


class MOSEACAgent(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            image_shape,
            gamma=0.99,
            hid_shape=(256, 256),
            a_lr=3e-4,
            c_lr=3e-4,
            batch_size=256,
            alpha=0.2,
            adaptive_alpha=True
    ):
        self.image_shape = (64, 64)
        self.actor = VanillaCNN_Actor(state_dim, self.image_shape, action_dim, hid_shape).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.q_critic = VanillaCNN_QCritic(state_dim, self.image_shape, action_dim, hid_shape).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        self.len_of_history_images = 4
        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = 0.005
        self.batch_size = batch_size

        self.alpha = alpha
        self.adaptive_alpha = adaptive_alpha
        if adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the SAC V2 paper
            self.target_entropy = torch.tensor(-action_dim, dtype=torch.float32, requires_grad=True, device=device)
            # We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
            self.log_alpha = torch.tensor(np.log(alpha), dtype=torch.float32, requires_grad=True, device=device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=c_lr)

    def select_action(self, state, image, deterministic, with_logprob=False):
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(device)
            image = torch.FloatTensor(image.reshape([1, self.len_of_history_images, self.image_shape[0], self.image_shape[1]])).to(device)
            a, _ = self.actor(state, image, deterministic, with_logprob)
        return a.cpu().numpy().flatten()

    def train(self, replay_buffer):
        s, imgs, a, r, s_prime, imgs_prime, dead_mask = replay_buffer.sample(self.batch_size)

        # Update Q Net
        with torch.no_grad():
            a_prime, log_pi_a_prime = self.actor(s_prime, imgs_prime)
            target_q1, target_q2 = self.q_critic_target(s_prime, imgs_prime, a_prime)
            target_q = torch.min(target_q1, target_q2)
            target_q = r + (1 - dead_mask) * self.gamma * (
                    target_q - self.alpha * log_pi_a_prime)  # Dead or Done is tackled by Randombuffer

        # Get current Q estimates
        current_q1, current_q2 = self.q_critic(s, imgs, a)
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        # Update Actor Net
        # Freeze Q-networks, so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for params in self.q_critic.parameters():
            params.requires_grad = False

        a, log_pi_a = self.actor(s, imgs)
        current_q1, current_q2 = self.q_critic(s, imgs, a)
        q = torch.min(current_q1, current_q2)
        a_loss = (self.alpha * log_pi_a - q).mean()
        self.actor_optimizer.zero_grad()
        a_loss.backward()
        self.actor_optimizer.step()

        for params in self.q_critic.parameters():
            params.requires_grad = True

        # Update alpha
        if self.adaptive_alpha:
            # we optimize log_alpha instead of alpha, which is aimed to force alpha = exp(log_alpha)> 0
            # if we optimize alpha directly, alpha might be < 0, which will lead to minium entropy.
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()

        # Update Target Net
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for params in self.q_critic.parameters():
            params.requires_grad = True

    def save(self, episode):
        torch.save(self.actor.state_dict(), save_path + "/moseac_actor{}.pth".format(episode))
        torch.save(self.q_critic.state_dict(), save_path + "/moseac_q_critic{}.pth".format(episode))
        print("current models has been saved")

    def load(self, episode):
        self.actor.load_state_dict(torch.load(load_path + "/moseac_actor{}.pth".format(episode)))
        self.q_critic.load_state_dict(torch.load(load_path + "/moseac_q_critic{}.pth".format(episode)))
        print("current models has been loaded")
