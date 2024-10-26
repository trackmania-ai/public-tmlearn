# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/agent_sac.ipynb.

# %% auto 0
__all__ = ['TwinnedCriticNet', 'SacAgent']

# %% ../../nbs/agent_sac.ipynb 1
import logging
from copy import deepcopy
from functools import cached_property, partial

import numpy as np
import torch
from fastai.optimizer import Adam
from fastcore.basics import store_attr

# %% ../../nbs/agent_sac.ipynb 3
class TwinnedCriticNet(torch.nn.Module):
    def __init__(self, critic_model_fn):
        super().__init__()
        self.critic1 = critic_model_fn()
        self.critic2 = critic_model_fn()

    def forward(self, state, action):
        return self.critic1(state, action), self.critic2(state, action)

# %% ../../nbs/agent_sac.ipynb 4
class SacAgent:
    tags = ["Sac"]
    """Require use of MixedActionHandler in Tmenv"""

    def __init__(
        self,
        actor_model,
        critic_model_fn,
        action_space,
        device="cpu",
        gamma=0.99,
        tau=0.005,
        optimizer_fn=partial(Adam, lr=1e-5, wd=1e-4),
    ):
        store_attr()
        self.actor = actor_model.to(device=device)
        self.actor.share_memory()

    @torch.no_grad()
    def select_action(self, state):
        self.actor.eval()
        action, _, _ = self.actor.sample(state)
        return tuple(action.detach().cpu().numpy()[0])

    @cached_property
    def actor_optimizer(self):
        return self.optimizer_fn(self.actor.parameters())

    @cached_property
    def critic(self):
        return TwinnedCriticNet(self.critic_model_fn).to(device=self.device)

    @cached_property
    def critic_optimizer(self):
        return self.optimizer_fn(self.critic.parameters())

    @cached_property
    def critic_target(self):
        critic_target = deepcopy(self.critic).eval()
        for p in critic_target.parameters():
            p.requires_grad = False
        return critic_target

    @cached_property
    def target_entropy(self):
        return -torch.prod(self.tensor(self.action_space.shape)).item()

    @cached_property
    def log_alpha(self):
        return torch.zeros(1, requires_grad=True, device=self.device)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @cached_property
    def alpha_optimizer(self):
        return self.optimizer_fn([self.log_alpha])

    @cached_property
    def nb_batch(self):
        return 0

    def set_lr(self, lr):
        for opt in [self.actor_optimizer, self.critic_optimizer, self.alpha_optimizer]:
            for g in opt.param_groups:
                g["lr"] = lr

    def tensor(self, data):
        return torch.tensor(data, device=self.device, dtype=torch.float)

    def _update_param(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _q_estimate(self, state, action):
        self.critic.train()
        q1, q2 = self.critic(state, self.tensor(action))
        return q1.squeeze(), q2.squeeze()

    @torch.no_grad()
    def _q_target(self, next_state, reward, done):
        self.actor.eval()
        action, log_prob, _ = self.actor.sample(next_state)
        q1, q2 = self.critic_target(next_state, action)
        q = (torch.min(q1, q2) - self.alpha * log_prob).squeeze()
        return self.tensor(reward) + (1 - self.tensor(done)) * self.gamma * q

    def _update_critic(self, state, next_state, action, reward, done, weights):
        q1, q2 = self._q_estimate(state, action)
        target_q = self._q_target(next_state, reward, done)
        errors = (torch.abs(q1 - target_q) + torch.abs(q2 - target_q)) / 2.0 + 1e-5
        errors = errors.detach()
        q1_loss = ((q1 - target_q).pow(2) * weights).mean()
        q2_loss = ((q2 - target_q).pow(2) * weights).mean()
        loss = (q1_loss + q2_loss) / 2
        self._update_param(self.critic_optimizer, loss)
        return loss, errors

    def _update_actor(self, state, weights):
        self.actor.train()
        action, log_prob, _ = self.actor.sample(state)
        with torch.no_grad():
            self.critic.eval()
            q1, q2 = self.critic(state, action)
            q = torch.min(q1, q2)

        loss = self.alpha * log_prob - q
        loss = (loss * weights).mean()
        self._update_param(self.actor_optimizer, loss)
        return loss, log_prob

    def _critic_soft_update(self):
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    def _update_alpha(self, log_prob, weights):
        loss = -self.log_alpha * (log_prob + self.target_entropy).detach()
        loss = (loss * weights).mean()
        self._update_param(self.alpha_optimizer, loss)
        return loss

    def fit_one_batch(self, batch, weights):
        state, next_state, action, reward, done = zip(*batch)

        critic_loss, errors = self._update_critic(
            state, next_state, action, reward, done, weights
        )
        actor_loss, log_prob = self._update_actor(state, weights)
        alpha_loss = self._update_alpha(log_prob, weights)
        self._critic_soft_update()

        metrics = dict(
            batch=self.nb_batch,
            mean_errors=errors.mean().item(),
            critic_loss=critic_loss.item(),
            actor_loss=actor_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha=self.alpha.item(),
        )

        self.nb_batch += 1
        return errors, metrics

    def state_dict(self):
        return dict(
            actor=self.actor.state_dict(),
            actor_optimizer=self.actor_optimizer.state_dict(),
            critic=self.critic.state_dict(),
            critic_optimizer=self.critic_optimizer.state_dict(),
            critic_target=self.critic_target.state_dict(),
            nb_batch=self.nb_batch,
        )

    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.actor.to(self.device)
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic.load_state_dict(state_dict["critic"])
        self.critic.to(self.device)
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.critic_target.load_state_dict(state_dict["critic_target"])
        self.critic_target.to(self.device)
        self.nb_batch = state_dict["nb_batch"]
