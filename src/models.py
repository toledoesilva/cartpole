"""
Defines the neural network architecture for the PPO agent.
"""

import torch.nn as nn


class ActorCritic(nn.Module):
    """
    Actor-Critic network with shared layers for feature extraction and separate heads for policy and value.
    """

    def __init__(self, input_dim: int, action_dim: int):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.policy_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        """
        Forward pass through the shared layers and separate heads.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Policy logits.
            torch.Tensor: State value.
        """
        features = self.shared(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features)
        return policy_logits, value
