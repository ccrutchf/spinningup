import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def weight_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data, np.sqrt(2))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class NoiseLinear(nn.Module):
    def __init__(self, in_size, out_size):
        super(NoiseLinear, self).__init__()
        self.linear = nn.Linear(in_size, out_size)
        self.sigma = 0.0

    def forward(self, x):

        if self.sigma > 0:
            noise_w = torch.randn_like(self.linear.weight).to(x.device) * self.sigma
            noise_b = torch.randn_like(self.linear.bias).to(x.device) * self.sigma
            x = F.linear(x, self.linear.weight + noise_w, self.linear.bias + noise_b)
            return x
        else:
            return self.linear(x)


class AtariCNN(nn.Module):
    def __init__(self, actions=4, dual=False):
        super(AtariCNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            Flatten(),
            NoiseLinear(64 * 7 * 7, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
        )

        self.action_head = NoiseLinear(512, actions)

        if dual:
            self.state_head = NoiseLinear(512, 1)
        else:
            self.state_head = None


    def forward(self, x):
        x = self.cnn(x)
        action_score = self.action_head(x)

        if self.state_head is not None:
            state_score = self.state_head(x)
            action_score = action_score - action_score.mean(dim=-1, keepdim=True)
            action_score = state_score + action_score

        return action_score

    def set_sigma(self, sigma):
        for m in self.modules():
            if isinstance(m, NoiseLinear):
                m.sigma = sigma

class MLP(nn.Module):
    def __init__(self, hidden_size=64, input_size=4, actions=4, dual=False):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            NoiseLinear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            NoiseLinear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )

        self.action_head = NoiseLinear(512, actions)

        if dual:
            self.state_head = NoiseLinear(512, 1)
        else:
            self.state_head = None

    def forward(self, x):
        x = self.mlp(x)
        action_score = self.action_head(x)

        if self.state_head is not None:
            state_score = self.state_head(x)
            action_score = action_score - action_score.mean(dim=-1, keepdim=True)
            action_score = state_score + action_score
        return action_score

    def set_sigma(self, sigma):
        for m in self.modules():
            if isinstance(m, NoiseLinear):
                m.sigma = sigma


class MujocoMLP(nn.Module):
    def __init__(self, hidden_size=64, input_size=4, output_size=4, output_act=None):
        super(MujocoMLP, self).__init__()

        self.mlp = nn.Sequential(
            NoiseLinear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            NoiseLinear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            NoiseLinear(hidden_size, output_size)
        )

        self.output_act = output_act

    def forward(self, x):
        x = self.mlp(x)
        if self.output_act is not None:
            x = F.tanh(x)
        return x

    def set_sigma(self, sigma):
        for m in self.modules():
            if isinstance(m, NoiseLinear):
                m.sigma = sigma