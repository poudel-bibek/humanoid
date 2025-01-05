import mujoco
import numpy as np
import time
from environment import SimpleEnv
from models import PolicyNetwork, ValueNetwork
from ppo import collect_trajectory, compute_returns_and_advantages, ppo_update
import torch.optim as optim

PATH_TO_MODEL = './mujoco_menagerie/unitree_g1/scene_with_hands.xml'

if __name__ == "__main__":
    # 1. Create model/data and environment
    model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
    data = mujoco.MjData(model)
    env = SimpleEnv(model, data)

    # 2. Instantiate policy and value networks
    obs_dim = env.observation_dim
    act_dim = env.action_dim
    policy = PolicyNetwork(obs_dim, act_dim)
    value_network = ValueNetwork(obs_dim)

    # 3. Create optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr=3e-4)
    value_optimizer = optim.Adam(value_network.parameters(), lr=1e-3)

    # Training hyperparameters
    num_iterations = 10     # number of outer training iterations
    horizon = 2048          # how many steps to collect per iteration
    gamma = 0.99
    clip_range = 0.2

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for iteration in range(num_iterations):
            # ----- Collect a trajectory -----
            trajectory = collect_trajectory(env, policy, value_network, horizon=horizon, gamma=gamma, viewer=viewer)
            
            # ----- Compute returns and advantages -----
            returns, advantages = compute_returns_and_advantages(trajectory, gamma=gamma)

            # ----- PPO update step (single epoch for simplicity) -----
            policy_loss, value_loss = ppo_update(
                policy,
                value_network,
                policy_optimizer,
                value_optimizer,
                trajectory,
                returns,
                advantages,
                clip_range=clip_range
            )
            
            print(f"Iter {iteration+1}/{num_iterations} | Policy Loss: {policy_loss:.3f} | Value Loss: {value_loss:.3f}")
    
    print("Training complete!")