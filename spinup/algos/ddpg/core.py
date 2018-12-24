import torch
import torch.nn.functional as F


import numpy as np
import copy

from spinup.utils.model import MujocoMLP
from spinup.utils.misc import RunningStat, AdaptiveParamNoiseSpec


class DDPGAgent:

    def __init__(self, obs_dim, act_dim, hidden_size, memory,
                 cuda=True,
                 param_noise_std=0,
                 action_noise_std=0.1,
                 gamma=0.99,
                 tau=0.01,
                 batch_size=64,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 clip_norm=None,
                 observation_range=(-5, 5),
                 action_range=(-1, 1)):

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_range = action_range
        self.clip_norm = clip_norm
        self.cuda = cuda

        self.memory = memory
        self.critic = MujocoMLP(hidden_size, obs_dim+act_dim, 1)
        self.actor = MujocoMLP(hidden_size, obs_dim, act_dim, output_act='tanh')

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        if cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()

        self.critic.eval()
        self.actor.eval()
        self.critic_target.eval()
        self.actor_target.eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.running_obs = RunningStat(observation_range)

        if param_noise_std > 0:
            self.param_noise = AdaptiveParamNoiseSpec(param_noise_std, param_noise_std)
        else:
            self.param_noise = None

        if action_noise_std > 0:
            self.action_noise = action_noise_std
        else:
            self.action_noise = None

    def train(self):
        batch = self.memory.sample(self.batch_size)

        self.critic.train()
        self.actor.train()

        obs0 = self.running_obs.normalize(batch['obs0'])
        obs1 = self.running_obs.normalize(batch['obs1'])
        obs0 = torch.from_numpy(obs0).float()
        obs1 = torch.from_numpy(obs1).float()

        terminals = torch.from_numpy(batch['terminals'])
        actions = torch.from_numpy(batch['actions'])
        rewards = torch.from_numpy(batch['rewards'])

        if self.cuda:
            obs0 = obs0.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            obs1 = obs1.cuda()
            terminals = terminals.cuda()

        action_value = self.actor(obs0)
        state_action_value = self.critic(torch.cat([obs0, actions], dim=-1))
        state_action_value_from_actor = self.critic(torch.cat([obs0, action_value], dim=-1))

        target_action_value = self.actor_target(obs1)
        target_state_action_value = self.critic_target(torch.cat([obs1, target_action_value], dim=-1))

        rhs = rewards + (1 - terminals) * self.gamma * target_state_action_value
        value_loss = F.mse_loss(state_action_value, rhs.detach())

        self.critic_optimizer.zero_grad()
        value_loss.backward()
        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm(self.critic.parameters(), self.clip_norm)
        self.critic_optimizer.step()

        policy_loss = 0 - state_action_value_from_actor.mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm(self.actor.parameters(), self.clip_norm)
        self.actor_optimizer.step()

        self.soft_sync()
        self.soft_sync()

        self.critic.eval()
        self.actor.eval()

        return value_loss.item(), policy_loss.item()

    def step(self, obs, noisy=False):
        obs = self.running_obs.normalize(obs)
        obs = torch.from_numpy(obs).float()

        if self.cuda:
            obs = obs.cuda()

        action = self.actor(obs)

        if self.param_noise is not None and noisy:

            self.actor.set_sigma(self.param_noise.current_stddev)
            action_noisy = self.actor(obs)
            self.actor.set_sigma(0)

            action = action_noisy

        q = self.critic(torch.cat([obs, action], dim=-1))
        q = q.detach().cpu().numpy()

        if self.action_noise is not None and noisy:
            noise = torch.randn_like(action) * self.action_noise
            if self.cuda:
                noise = noise.cuda()

            action += noise

        action = action.detach().cpu().numpy()
        action = np.clip(action, *self.action_range)

        return action, q, None, None

    def soft_sync(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def adapt_actor_param_noise(self):
        batch = self.memory.sample(self.batch_size)
        obs0 = self.running_obs.normalize(batch['obs0'])
        obs0 = torch.from_numpy(obs0).float()
        if self.cuda:
            obs0 = obs0.cuda()

        action = self.actor(obs0)
        self.actor.set_sigma(self.param_noise.current_stddev)
        action_noisy = self.actor(obs0)
        self.actor.set_sigma(0)

        dist = (action_noisy - action).pow(2).mean().sqrt().item()
        self.param_noise.adapt(dist)
        return dist


    def store(self, obs0, action, reward, obs1, terminal):
        self.memory.append(obs0, action, reward, obs1, terminal)
        self.running_obs.update(obs0)

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.param_noise is not None:
            self.param_noise.reset()