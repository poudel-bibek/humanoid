import os
import torch
import numpy as np
import mujoco
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import platform
import argparse
import glob
import time

from utils import plot_training_metrics
from environment import SimpleEnv
from models import PolicyNetwork, ValueNetwork, HybridPolicyNetwork
from ppo import collect_trajectory, compute_returns_and_advantages, ppo_update
from config import config as default_config

PATH_TO_MODEL = './mujoco_menagerie/unitree_g1/scene_with_hands.xml'

def evaluate_policy(env, policy, num_episodes=5, horizon=500, render=False, device='cpu'):
    """
    Evaluate the policy over a fixed number of episodes & horizon.
    Returns the average total reward.
    """
    policy.eval()
    returns = []

    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0.0

        for _ in range(horizon):
            # Move observation to policy's device
            obs_t = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                action_t, _ = policy.get_action(obs_t.unsqueeze(0))  # Policy will handle device placement
            action_np = action_t.cpu().squeeze(0).numpy()

            obs, reward, done, info = env.step(action_np)
            ep_reward += reward

            if done:
                break

        returns.append(ep_reward)

    policy.train()
    return np.mean(returns)

def load_saved_model(policy, value_net, model_dir, device):
    """
    Load the best saved policy and value network from the specified directory.
    """
    # Check if directory exists
    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory {model_dir} does not exist!")
    
    # Look for policy files
    policy_files = glob.glob(os.path.join(model_dir, "*policy*.pth"))
    value_files = glob.glob(os.path.join(model_dir, "*value*.pth"))
    
    if not policy_files:
        raise ValueError(f"No policy files found in {model_dir}")
    
    # Prefer 'best' policy if available, otherwise take the last one alphabetically
    policy_path = next((f for f in policy_files if 'best' in f.lower()), policy_files[-1])
    value_path = next((f for f in value_files if 'best' in f.lower()), value_files[-1])
    
    print(f"Loading policy from: {policy_path}")
    # Add weights_only=True to prevent the warning
    policy.load_state_dict(torch.load(policy_path, map_location=device, weights_only=True))
    
    if value_path:
        print(f"Loading value network from: {value_path}")
        # Add weights_only=True to prevent the warning
        value_net.load_state_dict(torch.load(value_path, map_location=device, weights_only=True))
    
    return policy, value_net

# ------------------- MAIN TRAINING LOOP ------------------- #
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train PPO agent')
    parser.add_argument('--platform', type=str, choices=['auto', 'macos', 'linux'], 
                       default='auto', help='Platform-specific viewer (auto, macos, or linux)')
    parser.add_argument('--render', action='store_true', help='Enable visualization')
    parser.add_argument('--render-mode', type=str, choices=['window', 'offscreen'], 
                       default='window', help='Rendering mode (window or offscreen)')
    # Add GPU argument
    parser.add_argument('--use_gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--evaluate', action='store_true', help='Run in evaluation mode only')
    parser.add_argument('--policy_path', type=str, default='./models', 
                       help='Path to folder containing policy models')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file')
    args = parser.parse_args()

    # Load config
    config = default_config.copy()  # Make a copy of default config
    if args.config:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", args.config)
        custom_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_config)
        # Update default config with custom values
        config.update(custom_config.config)

    # Device setup
    if args.use_gpu:
        if platform.system() == 'Darwin':  # macOS
            if torch.backends.mps.is_available():
                device = torch.device('mps')
                print("Using Apple Metal GPU (MPS)")
            else:
                print("MPS not available. Falling back to CPU.")
                device = torch.device('cpu')
        else:  # Linux/Windows
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("CUDA not available. Using CPU.")
                device = torch.device('cpu')
    else:
        device = torch.device('cpu')
        print("Using CPU as specified.")

    PATH_TO_MODEL = './mujoco_menagerie/unitree_g1/scene_with_hands.xml'

    # Create model/data and environment
    model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
    data = mujoco.MjData(model)

    env = SimpleEnv(
        model, 
        data, 
        forward_reward_weight=config['forward_reward_weight'],
        ctrl_cost_weight=config['ctrl_cost_weight'],
        healthy_reward=config['healthy_reward']
    )

    # Create Policy and Value networks
    obs_dim = env.observation_dim
    act_dim = env.action_dim
    policy = HybridPolicyNetwork(
        obs_dim, 
        act_dim, 
        hidden_size=config['policy_hidden_size'],
        sequence_length=config['transformer_sequence_length']
    ).to(device)
    value_network = ValueNetwork(
        obs_dim, 
        hidden_size=config['value_hidden_size']
    ).to(device)

    # Optimizers
    policy_optimizer = optim.Adam(policy.parameters(), lr=config['policy_lr'])
    value_optimizer = optim.Adam(value_network.parameters(), lr=config['value_lr'])

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
        forward_reward_weight=config['forward_reward_weight'],
        ctrl_cost_weight=config['ctrl_cost_weight'],
        healthy_reward=config['healthy_reward']
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

    # If in evaluation mode, load the saved model and run evaluation
    if args.evaluate:
        print("Running in evaluation mode...")
        policy, value_network = load_saved_model(
            policy, value_network, args.policy_path, device
        )
        
        # Initialize viewer for evaluation
        viewer = None
        if args.render:
            if args.platform == 'macos' or (args.platform == 'auto' and platform.system() == 'Darwin'):
                viewer = mujoco.viewer.launch_passive(model, data)
            else:
                try:
                    import mujoco_viewer
                    viewer = mujoco_viewer.MujocoViewer(
                        model, 
                        data,
                        mode=args.render_mode,
                        title="Humanoid Evaluation"
                    )
                except ImportError:
                    print("Please install mujoco-python-viewer for visualization")
        
        try:
            # Run multiple evaluation episodes
            num_eval_episodes = 10
            eval_horizon = 1000  # Longer horizon for evaluation
            
            total_rewards = []
            max_distance = 0
            
            for ep in range(num_eval_episodes):
                obs = env.reset()
                ep_reward = 0
                
                for step in range(eval_horizon):
                    # Get action from policy
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        action_t, _ = policy.get_action(obs_t.unsqueeze(0))
                    action = action_t.cpu().squeeze(0).numpy()
                    
                    # Step environment
                    obs, reward, done, _ = env.step(action)
                    ep_reward += reward
                    
                    # Update visualization
                    if viewer is not None:
                        if hasattr(viewer, 'sync'):
                            viewer.sync()
                        elif hasattr(viewer, 'render'):
                            viewer.render()
                        time.sleep(0.01)
                    
                    if done:
                        break
                
                # Record metrics
                total_rewards.append(ep_reward)
                current_distance = env.metrics_history["distance_from_origin"][-1]
                max_distance = max(max_distance, current_distance)
                
                print(f"Episode {ep + 1}/{num_eval_episodes} - "
                      f"Reward: {ep_reward:.2f} - "
                      f"Distance: {current_distance:.2f}m")
            
            # Print summary statistics
            print("\nEvaluation Summary:")
            print(f"Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
            print(f"Max Distance Achieved: {max_distance:.2f}m")
            
        finally:
            # Clean up viewer
            if viewer is not None and hasattr(viewer, 'close'):
                viewer.close()
        
        return  # Exit after evaluation
    
    try:
        # Initialize viewer based on settings
        if not args.render:
            print("Running without visualization (default mode)...")
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
        for iteration in range(config['num_iterations']):

            # 1) Collect a trajectory
            trajectory = collect_trajectory(
                env, policy, value_network,
                horizon=config['horizon'], gamma=config['gamma'],
                viewer=viewer, device=device
            )

            # 2) Compute returns & advantages
            returns, advantages = compute_returns_and_advantages(
                trajectory, gamma=config['gamma']
            )

            # 3) PPO update
            policy_loss, value_loss = ppo_update(
                policy,
                value_network,
                policy_optimizer,
                value_optimizer,
                trajectory,
                returns,
                advantages,
                clip_range=config['clip_range'],
                n_epochs=config['ppo_epochs'],
                batch_size=config['batch_size']
            )

            # Logging: average reward in the last horizon steps
            if len(env.metrics_history["total_reward"]) >= config['horizon']:
                avg_reward = np.mean(env.metrics_history["total_reward"][-config['horizon']:])
            else:
                avg_reward = np.mean(env.metrics_history["total_reward"])

            avg_reward_per_iter.append(avg_reward)

            # Distance from origin (last step recorded in the environment)
            dist_from_origin = env.metrics_history["distance_from_origin"][-1]
            dist_from_origin_iter.append(dist_from_origin)

            print(f"Iter {iteration+1}/{config['num_iterations']} | "
                  f"Policy Loss: {policy_loss:.3f} | "
                  f"Value Loss: {value_loss:.3f} | "
                  f"Avg Reward (train): {avg_reward:.3f}")

            # Evaluate periodically
            if (iteration + 1) % config['eval_every'] == 0:
                eval_reward = evaluate_policy(
                    test_env, 
                    policy, 
                    num_episodes=config['eval_episodes'],
                    horizon=config['eval_horizon']
                )
                print(f"Evaluation Reward: {eval_reward:.3f}")
                
                # Save if best
                if eval_reward > best_avg_reward:
                    best_avg_reward = eval_reward
                    torch.save(
                        policy.state_dict(), 
                        os.path.join(config['models_dir'], "best_policy.pth"),
                        _use_new_zipfile_serialization=True
                    )
                    torch.save(
                        value_network.state_dict(), 
                        os.path.join(config['models_dir'], "best_value.pth"),
                        _use_new_zipfile_serialization=True
                    )
                    print(f"New best model saved with eval_reward={eval_reward:.3f}")
                
            # Save periodic checkpoints
            if (iteration + 1) % config['save_every'] == 0:
                checkpoint_path = os.path.join(
                    config['models_dir'], 
                    f"checkpoint_{iteration+1}"
                )
                os.makedirs(checkpoint_path, exist_ok=True)
                torch.save(policy.state_dict(), 
                         os.path.join(checkpoint_path, "policy.pth"))
                torch.save(value_network.state_dict(), 
                         os.path.join(checkpoint_path, "value.pth"))

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

    # Plot training metrics
    plot_training_metrics(avg_reward_per_iter, dist_from_origin_iter)

    # -------------- Final Save (if you want) --------------
    # Save final networks (not necessarily best)
    torch.save(
        policy.state_dict(), 
        "./models/final_policy.pth",
        _use_new_zipfile_serialization=True
    )
    torch.save(
        value_network.state_dict(), 
        "./models/final_value.pth",
        _use_new_zipfile_serialization=True
    )
    print("Final models saved to disk.")

# Entry point
if __name__ == "__main__":
    main()