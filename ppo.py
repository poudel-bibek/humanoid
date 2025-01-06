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
               clip_range=0.2, value_coef=0.5, entropy_coef=0.0):
    """
    Perform one epoch of PPO updates over the entire trajectory data.
    In a real PPO, you'd break the data into minibatches and do multiple epochs.
    """
    states = trajectory["states"]
    actions = trajectory["actions"]
    old_log_probs = trajectory["log_probs"]

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # New log_probs
    new_log_probs = policy.log_prob(states, actions)
    ratio = torch.exp(new_log_probs - old_log_probs)  # shape (T,)

    # Clipped surrogate objective
    obj1 = ratio * advantages
    obj2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    policy_loss = -torch.min(obj1, obj2).mean()

    # Value loss
    values_pred = value_network(states)
    value_loss = nn.MSELoss()(values_pred, returns)

    # Optional entropy bonus
    mean, std = policy.forward(states)
    dist = torch.distributions.Normal(mean, std)
    entropy = dist.entropy().sum(-1).mean()

    # Combine
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

    # Optimize
    optimizer_policy.zero_grad()
    optimizer_value.zero_grad()
    total_loss.backward()
    optimizer_policy.step()
    optimizer_value.step()

    return policy_loss.item(), value_loss.item()

