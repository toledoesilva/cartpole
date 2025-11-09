"""
PPOAgent class implementing the Proximal Policy Optimization algorithm.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from src.models import ActorCritic


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for training and interacting with the environment.
    """

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        policy_lr: float = 3e-4,
        value_lr: float = 1e-3,
        clip_epsilon: float = 0.2,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(input_dim, action_dim).to(self.device)
        self.policy_optimizer = optim.Adam(
            self.model.policy_head.parameters(), lr=policy_lr
        )
        self.value_optimizer = optim.Adam(
            self.model.value_head.parameters(), lr=value_lr
        )
        self.loss_fn = nn.MSELoss()

    def compute_advantages(self, rewards, values, dones):
        """
        Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            rewards (np.ndarray): Rewards from the environment.
            values (np.ndarray): Value estimates.
            dones (np.ndarray): Done flags.

        Returns:
            np.ndarray: Computed advantages.
        """
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        return advantages

    def update(self, trajectories):
        """
        Update the policy and value networks using the collected trajectories.

        Args:
            trajectories (dict): Collected trajectories containing states, actions, rewards, etc.
        """
        # convert list of states (possibly numpy arrays) to a single ndarray first
        states = torch.tensor(np.array(trajectories["states"]), dtype=torch.float32).to(
            self.device
        )
        actions = torch.tensor(trajectories["actions"], dtype=torch.int64).to(
            self.device
        )
        old_log_probs = torch.tensor(trajectories["log_probs"], dtype=torch.float32).to(
            self.device
        )
        returns = torch.tensor(trajectories["returns"], dtype=torch.float32).to(
            self.device
        )
        advantages = torch.tensor(trajectories["advantages"], dtype=torch.float32).to(
            self.device
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy update: run a forward pass for policy gradients
        logits, _values_for_policy = self.model(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)

        surrogate1 = ratio * advantages
        surrogate2 = (
            torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
            * advantages
        )
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Value update: recompute values with a fresh forward to get a new autograd graph
        _logits_for_value, values = self.model(states)
        value_loss = self.loss_fn(values.squeeze(), returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
