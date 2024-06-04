import torch
import torch.nn as nn
import torch.onnx
import torch.nn.functional as F
from torch.distributions import Normal
import os

current_path = os.getcwd()


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


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape, h_acti=nn.ReLU, o_acti=nn.ReLU):
        super(Actor, self).__init__()
        layers = [state_dim] + list(hid_shape)
        self.action_dim = action_dim
        self.a_net = build_net(layers, h_acti, o_acti)
        self.mu_layer = nn.Linear(layers[-1], action_dim)
        self.log_std_layer = nn.Linear(layers[-1], action_dim)

        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state):
        # Network with Enforcing Action Bounds
        net_out = self.a_net(state)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mu, std)
        u = mu
        a_t = F.relu6(u[:, 0]).unsqueeze(1)
        a_m = torch.tanh(u[:, 1:3])
        a = torch.cat([a_t, a_m], dim=1)
        logp_pi_a = None
        return a, logp_pi_a


index = 'Change to your model id'  # select the model
state_sim = 49  # all we need to adjust is here
action_dim = 3
hid_shape = (256, 256, 256)

model = Actor(state_sim, action_dim, hid_shape)
model.load_state_dict(torch.load("model/moseac_actor{}.pth".format(index)))
model.eval()

fake_input = torch.rand(1, 49)
torch.onnx.export(model, fake_input, current_path + "/model/model.onnx", verbose=True, input_names=['state'], output_names=['action'], opset_version=13)
print("pth model has been transformed to a onxx file")
