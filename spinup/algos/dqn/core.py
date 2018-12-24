import torch
import torch.nn.functional as F

import copy
import numpy as np

from spinup.utils.model import AtariCNN, MLP

class DQNAgent:
    def __init__(self, obs_shape, action_space, lr, dual, cuda=True):
        if len(obs_shape) == 4:
            self.q_model = AtariCNN(action_space.n, dual=dual)
            self.scale = 255.0
        else:
            self.q_model = MLP(input_size=obs_shape[-1], actions=action_space.n, dual=dual)
            self.scale = 1.0

        self.action_space = action_space
        self.target_q_model = copy.deepcopy(self.q_model)
        self.cuda = cuda

        if self.cuda:
            self.q_model.cuda()
            self.target_q_model.cuda()

        self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=lr)

    def step(self, obs, epsilon):
        if np.random.rand() > epsilon:
            obs = np.array([obs])
            if obs.ndim == 4:
                obs = obs.transpose(0, 3, 1, 2) / self.scale
            obs = torch.from_numpy(obs).float()
            if self.cuda:
                obs = obs.cuda()
            q = self.q_model(obs)
            qval = q.max().item()
            action = q.argmax().item()
            return action, qval
        else:
            return self.action_space.sample(), None

    def one_hot(self, y):
        y_onehot = torch.zeros(y.size(0), self.action_space.n)
        if self.cuda:
            y_onehot = y_onehot.cuda()
        y_onehot.scatter_(1, y.view(-1, 1).long(), 1)
        return y_onehot

    def update(self, inputs, double_q=True, gamma=0.99, grad_norm_clip=10):
        inputs = [torch.from_numpy(np.array(x, dtype=np.float32)) for x in inputs]
        if self.cuda:
            inputs = [x.cuda() for x in inputs]
        obses_t, actions, rewards, obses_tp1, dones, weights = inputs

        if obses_t.ndimension() == 4:
            obses_t = obses_t.permute(0, 3, 1, 2) / self.scale
            obses_tp1 = obses_tp1.permute(0, 3, 1, 2) / self.scale

        q_t = self.q_model(obses_t)
        q_tp1 = self.target_q_model(obses_tp1)
        q_t_selected = torch.sum(q_t * self.one_hot(actions), dim=1)

        if double_q:
            q_tp1_using_online_net = self.q_model(obses_tp1)
            q_tp1_best_using_online_net = q_tp1_using_online_net.max(dim=1)[1]
            q_tp1_best = torch.sum(q_tp1 * self.one_hot(q_tp1_best_using_online_net), dim=1)
        else:
            q_tp1_best = q_tp1.max(dim=1)[0]

        rhs = rewards.squeeze() + gamma * (1 - dones) * q_tp1_best.detach()
        td_err = q_t_selected.detach() - rhs.detach()

        loss = F.smooth_l1_loss(q_t_selected, rhs, reduction='none')
        loss_weighted = torch.mean(loss * weights)

        self.optimizer.zero_grad()
        loss_weighted.backward()
        torch.nn.utils.clip_grad_norm_(self.q_model.parameters(), grad_norm_clip)
        self.optimizer.step()

        return td_err.detach().cpu().numpy(), loss.mean().item()

    def sync(self):
        self.target_q_model = copy.deepcopy(self.q_model)

    def save(self, model_file):
        data = {
            'q_model': self.q_model.state_dict(),
            'target_model': self.target_q_model.state_dict()}
        torch.save(data, model_file)

    def load(self, model_file):
        data = torch.load(model_file)
        self.q_model.load_state_dict(data['q_model'])
        self.target_q_model.load_state_dict(data['target_model'])
