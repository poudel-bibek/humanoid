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

class TransformerMovementPlanner(nn.Module):
    def __init__(self, state_dim, sequence_length=32, nhead=8, num_layers=3, dim_feedforward=512):
        super().__init__()
        
        # Embedding layer to project state to transformer dimension
        self.state_embedding = nn.Linear(state_dim, dim_feedforward)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.zeros(sequence_length, dim_feedforward))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_feedforward,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(dim_feedforward, state_dim)
        
        self.sequence_length = sequence_length
        
    def forward(self, state_history):
        # state_history shape: (batch_size, sequence_length, state_dim)
        
        # Embed states
        x = self.state_embedding(state_history)
        
        # Add positional encoding
        x = x + self.pos_encoder.unsqueeze(0)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Project back to state space
        future_states = self.output_proj(x)
        
        return future_states

class HybridPolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256, sequence_length=32):
        super().__init__()
        
        # Original policy network components
        self.fc = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_size),  # *2 because we'll concatenate with transformer output
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        
        # Add transformer planner
        self.transformer = TransformerMovementPlanner(
            state_dim=obs_dim,
            sequence_length=sequence_length
        )
        
        # State history buffer - now we'll maintain separate histories for each batch item
        self.state_histories = {}  # Dictionary to store histories for different batch sizes
        self.sequence_length = sequence_length
        
    def get_state_history(self, batch_size, obs_dim):
        """Get or create state history buffer for the given batch size"""
        if batch_size not in self.state_histories:
            self.state_histories[batch_size] = torch.zeros(
                batch_size, self.sequence_length, obs_dim,
                device=next(self.parameters()).device  # Use same device as model
            )
        return self.state_histories[batch_size]
        
    def update_state_history(self, state):
        """Update state history handling both batched and unbatched inputs"""
        # Ensure state is 2D and on correct device
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Move state to same device as model if needed
        if state.device != next(self.parameters()).device:
            state = state.to(next(self.parameters()).device)
        
        batch_size, obs_dim = state.shape
        state_history = self.get_state_history(batch_size, obs_dim)
        
        # Roll and update
        state_history = torch.roll(state_history, shifts=-1, dims=1)
        state_history[:, -1] = state
        
        # Store updated history
        self.state_histories[batch_size] = state_history
        
        return state_history
        
    def forward(self, obs):
        # Ensure obs is 2D and on correct device
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        # Move obs to model's device if needed
        device = next(self.parameters()).device
        obs = obs.to(device)
            
        # Update and get state history
        state_history = self.update_state_history(obs)
        
        # Get future state predictions from transformer
        future_states = self.transformer(state_history)
        next_state_pred = future_states[:, -1]  # Take the last predicted state
        
        # Ensure both tensors are on same device before concatenating
        next_state_pred = next_state_pred.to(device)
        
        # Comment out debug prints
        # print(f"Debug - Forward pass devices:")
        # print(f"obs device: {obs.device}")
        # print(f"next_state_pred device: {next_state_pred.device}")
        
        # Concatenate current state with predicted next state
        combined_features = torch.cat([obs, next_state_pred], dim=-1)
        
        # Forward through policy network
        mean = self.fc(combined_features)
        std = torch.exp(self.log_std)
        
        return mean, std

    def get_action(self, obs):
        """
        Sample an action from the policy's Gaussian distribution,
        and return the log probability of that action.
        """
        # Ensure obs is on correct device
        device = next(self.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
            
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action, log_prob

    def log_prob(self, obs, action):
        """
        Given states and actions, compute the log probability under this policy.
        """
        # Ensure inputs are on correct device
        device = next(self.parameters()).device
        if obs.device != device:
            obs = obs.to(device)
        if action.device != device:
            action = action.to(device)
            
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(axis=-1)