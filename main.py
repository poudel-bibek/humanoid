import os
import torch
import numpy as np
import mujoco
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import platform
import argparse


from environment import SimpleEnv
from models import PolicyNetwork, ValueNetwork, HybridPolicyNetwork
from ppo import collect_trajectory, compute_returns_and_advantages, ppo_update

PATH_TO_MODEL = './mujoco_menagerie/unitree_g1/scene_with_hands.xml'

def evaluate_policy(env, policy, num_episodes=5, horizon=500, render=False):
    """
    Evaluate the policy over a fixed number of episodes & horizon.
    Returns the average total reward.
    """
    policy.eval()  # put in eval mode
    returns = []

    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0.0

        for _ in range(horizon):
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_t, _ = policy.get_action(obs_t)
            action_np = action_t.squeeze(0).numpy()

            obs, reward, done, info = env.step(action_np)
            ep_reward += reward

            if done:
                break

            if render:
                # Rendering if desired
                pass

        returns.append(ep_reward)

    policy.train()  # back to train mode
    return np.mean(returns)


# ------------------- MAIN TRAINING LOOP ------------------- #
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--platform', type=str, choices=['auto', 'macos', 'linux'], 
                       default='auto', help='Platform-specific viewer (auto, macos, or linux)')
    parser.add_argument('--no-render', action='store_true', help='Disable visualization')
    parser.add_argument('--render-mode', type=str, choices=['window', 'offscreen'], 
                       default='window', help='Rendering mode (window or offscreen)')
    args = parser.parse_args()

    PATH_TO_MODEL = './mujoco_menagerie/unitree_g1/scene_with_hands.xml'

    # Create model/data and environment
    model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
    data = mujoco.MjData(model)

    env = SimpleEnv(
        model, 
        data, 
        forward_reward_weight=2.0,    # Increased to emphasize forward movement
        ctrl_cost_weight=0.005,       # Reduced to allow more exploration
        healthy_reward=1.5            # Increased to emphasize stability
    )

    # Create Policy and Value networks
    obs_dim = env.observation_dim
    act_dim = env.action_dim
    policy = HybridPolicyNetwork(obs_dim, act_dim, hidden_size=256)
    value_network = ValueNetwork(obs_dim, hidden_size=256)

    # Optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr=1e-4)
    value_optimizer = optim.Adam(value_network.parameters(), lr=5e-4)

    # PPO hyperparameters
    num_iterations = 1000       # Increased from 500
    horizon = 2048             # Keep same
    gamma = 0.99              # Keep same
    clip_range = 0.2          # Keep same

    # Evaluation frequency
    eval_every = 10          # Evaluate every 10 iterations to track progress

    # Tracking for plotting
    avg_reward_per_iter = []
    dist_from_origin_iter = []

    # Track the best reward to save the best model
    best_avg_reward = -np.inf

    # We can optionally do a separate test environment for evaluation,
    # but here we just re-use the same environment (this is simplistic).
    test_env = SimpleEnv(
        model,
        data,
        forward_reward_weight=2.0,
        ctrl_cost_weight=0.005,
        healthy_reward=1.5
    )

    if not os.path.exists("./models"):
        os.makedirs("./models")
    if not os.path.exists("./plots"):
        os.makedirs("./plots")

    # Initialize viewer as None
    viewer = None
    
    # Determine platform for viewer type
    if args.platform == 'auto':
        use_macos_viewer = platform.system() == 'Darwin'
    else:
        use_macos_viewer = args.platform == 'macos'

    try:
        # Initialize viewer based on settings
        if args.no_render:
            print("Running without visualization as requested...")
        else:
            if use_macos_viewer:
                # Use the built-in MuJoCo viewer for macOS
                try:
                    viewer = mujoco.viewer.launch_passive(model, data)
                except Exception as e:
                    print(f"Failed to initialize macOS viewer: {e}")
                    print("Running without visualization...")
            else:
                # Use mujoco-python-viewer for other platforms
                try:
                    import mujoco_viewer
                    viewer = mujoco_viewer.MujocoViewer(
                        model, 
                        data,
                        mode=args.render_mode,
                        title="Humanoid Training",
                        hide_menus=False
                    )
                except ImportError:
                    print("Please install mujoco-python-viewer: pip install mujoco-python-viewer")
                    return
                except Exception as e:
                    print(f"Error initializing viewer: {e}")
                    print("Running without visualization...")

        # Main training loop
        for iteration in range(num_iterations):
            # 1) Collect a trajectory
            trajectory = collect_trajectory(env, policy, value_network,
                                         horizon=horizon, gamma=gamma, viewer=viewer)

            # 2) Compute returns & advantages
            returns, advantages = compute_returns_and_advantages(trajectory, gamma=gamma)

            # 3) PPO update
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

            # Logging: average reward in the last horizon steps
            if len(env.metrics_history["total_reward"]) >= horizon:
                avg_reward = np.mean(env.metrics_history["total_reward"][-horizon:])
            else:
                avg_reward = np.mean(env.metrics_history["total_reward"])

            avg_reward_per_iter.append(avg_reward)

            # Distance from origin (last step recorded in the environment)
            dist_from_origin = env.metrics_history["distance_from_origin"][-1]
            dist_from_origin_iter.append(dist_from_origin)

            print(f"Iter {iteration+1}/{num_iterations} | "
                  f"Policy Loss: {policy_loss:.3f} | "
                  f"Value Loss: {value_loss:.3f} | "
                  f"Avg Reward (train): {avg_reward:.3f}")

            # ---------------- Evaluate the current policy ----------------
            eval_reward = evaluate_policy(test_env, policy, num_episodes=3, horizon=500)
            print(f"Evaluation Reward: {eval_reward:.3f}")

            # If the eval reward is the best so far, save the model
            if eval_reward > best_avg_reward:
                best_avg_reward = eval_reward
                torch.save(policy.state_dict(), "./models/best_policy.pth")
                torch.save(value_network.state_dict(), "./models/best_value.pth")
                print(f"New best model saved with eval_reward={eval_reward:.3f}")

            # Check if viewer is still alive (for non-macOS viewer)
            if not use_macos_viewer and viewer is not None:
                if hasattr(viewer, 'is_alive') and not viewer.is_alive:
                    print("Viewer window was closed. Stopping training.")
                    break

    finally:
        # Clean up the viewer
        if not use_macos_viewer and viewer is not None:
            if hasattr(viewer, 'close'):
                viewer.close()

    print("Training complete!")

    # ---------- Plot some training metrics ----------
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(avg_reward_per_iter, label='Avg Reward (Train)')
    plt.title('Reward Progress (Training)')
    plt.xlabel('Iteration')
    plt.ylabel('Avg Reward')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(dist_from_origin_iter, label='Distance from Origin', color='orange')
    plt.title('Distance from Origin')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.legend()

    plt.tight_layout()
    plt.savefig("./plots/training_metrics.png")

    # -------------- Final Save (if you want) --------------
    # Save final networks (not necessarily best)
    torch.save(policy.state_dict(), "./models/final_policy.pth")
    torch.save(value_network.state_dict(), "./models/final_value.pth")
    print("Final models saved to disk.")

# Entry point
if __name__ == "__main__":
    main()