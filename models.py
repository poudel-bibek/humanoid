import torch
import torch.nn as nn

# ----- 1) Define the Policy Network (Gaussian) -----
class PolicyNetwork(nn.Module):
    """
    A simple MLP policy that outputs a Gaussian distribution over actions:
        mean = MLP(state)
        log_std = learned parameter (or MLP output)
    """
    def __init__(self, obs_dim, act_dim, hidden_size=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
        # We'll keep log_std as a trainable parameter (per-dimension or single scalar).
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs):
        """
        Forward pass: produce the mean for the Gaussian.
        We also have a learnable log_std we add.
        """
        mean = self.fc(obs)
        std = torch.exp(self.log_std)
        return mean, std

    def get_action(self, obs):
        """
        Sample an action from the policy's Gaussian distribution,
        and return the log probability of that action.
        """
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob

    def log_prob(self, obs, action):
        """
        Given states and actions, compute the log probability under this policy.
        """
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(axis=-1)

# ----- 2) Define the Value Network -----
class ValueNetwork(nn.Module):
    """
    A simple MLP for state-value function.
    """
    def __init__(self, obs_dim, hidden_size=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, obs):
        return self.fc(obs).squeeze(-1)  # shape: (batch,)