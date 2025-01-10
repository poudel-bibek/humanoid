import time
import torch
import torch.nn as nn

# ------------------- PPO HELPER FUNCTIONS ------------------- #
def collect_trajectory(env, policy, value_network, horizon=2048, gamma=0.99, viewer=None):
    """
    Run the policy in the environment for 'horizon' timesteps.
    Store (state, action, log_prob, reward, value) at each step.
    """
    states = []
    actions = []
    log_probs = []
    rewards = []
    values = []

    obs = env.reset()
    done = False

    for t in range(horizon):
        obs_t = torch.as_tensor(obs, dtype=torch.float32)

        # Value estimate
        value_t = value_network(obs_t.unsqueeze(0))

        # Sample action
        with torch.no_grad():
            action_t, log_prob_t = policy.get_action(obs_t.unsqueeze(0))

        action_np = action_t.squeeze(0).numpy()
        next_obs, reward, done, info = env.step(action_np)

        states.append(obs_t)
        actions.append(action_t.squeeze(0))
        log_probs.append(log_prob_t.squeeze(0))
        rewards.append(reward)
        values.append(value_t.item())

        obs = next_obs
        if done:
            break

        if viewer is not None:
            viewer.sync()  # update rendering
            time.sleep(0.01)

    trajectory = {
        "states": torch.stack(states),
        "actions": torch.stack(actions),
        "log_probs": torch.stack(log_probs),
        "rewards": torch.tensor(rewards, dtype=torch.float32),
        "values": torch.tensor(values, dtype=torch.float32),
    }
    return trajectory


def compute_returns_and_advantages(trajectory, gamma=0.99):
    """
    For each step t, compute:
      G_t = r_t + gamma*r_{t+1} + ...  (discounted returns)
    and advantage estimates:
      A_t = G_t - V(s_t)
    
    This is the simplest version (no GAE-lambda), but we show a structure
    for lam if desired.
    
    Returns the final Tensors for 'returns' and 'advantages'.
    """
    rewards = trajectory["rewards"]
    values = trajectory["values"]
    length = len(rewards)

    returns = torch.zeros(length, dtype=torch.float32)
    advantages = torch.zeros(length, dtype=torch.float32)

    # Discounted returns
    running_return = 0.0
    for t in reversed(range(length)):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return

    # Advantage: returns - values
    advantages = returns - values
    return returns, advantages


def ppo_update(policy, value_network, optimizer_policy, optimizer_value,
               trajectory, returns, advantages,
               clip_range=0.2, value_coef=0.5, entropy_coef=0.01):
    """
    Perform one epoch of PPO updates over the entire trajectory data.
    Enhanced with better clipping and entropy bonus for exploration.
    """
    states = trajectory["states"]
    actions = trajectory["actions"]
    old_log_probs = trajectory["log_probs"]

    # Normalize advantages (important for training stability)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Multiple epochs of PPO updates (important for sample efficiency)
    n_epochs = 4
    batch_size = 64
    n_samples = len(states)
    
    total_policy_loss = 0
    total_value_loss = 0
    
    for _ in range(n_epochs):
        # Random permutation for batching
        indices = torch.randperm(n_samples)
        
        for start_idx in range(0, n_samples, batch_size):
            # Get mini-batch
            batch_indices = indices[start_idx:start_idx + batch_size]
            states_batch = states[batch_indices]
            actions_batch = actions[batch_indices]
            old_log_probs_batch = old_log_probs[batch_indices]
            advantages_batch = advantages[batch_indices]
            returns_batch = returns[batch_indices]

            # New log_probs
            new_log_probs = policy.log_prob(states_batch, actions_batch)
            ratio = torch.exp(new_log_probs - old_log_probs_batch)

            # Clipped surrogate objective
            obj1 = ratio * advantages_batch
            obj2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages_batch
            policy_loss = -torch.min(obj1, obj2).mean()

            # Value loss with clipping (PPO2 style)
            values_pred = value_network(states_batch)
            values_clipped = trajectory["values"][batch_indices] + \
                           torch.clamp(values_pred - trajectory["values"][batch_indices],
                                     -clip_range, clip_range)
            v_loss1 = (values_pred - returns_batch).pow(2)
            v_loss2 = (values_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

            # Entropy bonus for exploration
            mean, std = policy.forward(states_batch)
            dist = torch.distributions.Normal(mean, std)
            entropy = dist.entropy().sum(-1).mean()

            # Combined loss
            total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            # Optimize
            optimizer_policy.zero_grad()
            optimizer_value.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(value_network.parameters(), max_norm=0.5)
            
            optimizer_policy.step()
            optimizer_value.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

    # Return average losses
    n_updates = n_epochs * (n_samples // batch_size)
    return total_policy_loss / n_updates, total_value_loss / n_updates

