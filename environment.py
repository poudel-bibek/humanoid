import mujoco
import numpy as np
from collections import deque

class SimpleEnv:
    def __init__(self, model, data, forward_reward_weight=2.0, ctrl_cost_weight=0.005, healthy_reward=1.0):
        """
        A simple environment that calculates a more complex reward.

        Args:
            model (MjModel): The MuJoCo model.
            data (MjData): The MuJoCo data associated with model.
            forward_reward_weight (float): Weight for forward velocity reward (increased from 1.0)
            ctrl_cost_weight (float): Penalty for control actions (kept small to encourage exploration)
            healthy_reward (float): Reward for staying healthy/stable
        """
        # Use the same model/data from outside
        self.model = model
        self.data = data

        # Dimensions
        self.action_dim = self.model.nu
        self.observation_dim = self.model.nq + self.model.nv

        # Hyperparameters for reward function
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.healthy_reward = healthy_reward

        # Additional reward components
        self.stability_cost_weight = 0.1  # New: penalize unstable poses
        self.progress_bonus = 0.5  # New: bonus for consistent forward progress

        # For logging
        self.metrics_history = {
            "forward_reward": [],
            "reward_quadctrl": [],
            "reward_alive": [],
            "distance_from_origin": [],
            "total_reward": []
        }

        self.last_action = np.zeros_like(self.data.ctrl)

        # Add to initialization
        self.reward_history = deque(maxlen=1000)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.last_action = np.zeros_like(self.data.ctrl)
        return self._get_obs()

    def step(self, action):
        """
        Step the environment forward with the given action.
        Reward components:
            # Move forward efficiently
            # Maintain stability (stay upright/healthy)
            # Use minimal control effort (be energy efficient)
        Returns: obs, total_reward, done, info_dict
        """
        # Apply control
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # Compute observations
        obs = self._get_obs()

        # ------- REWARD COMPONENTS ------- 
        # 1. Forward velocity (just an example: velocity along x-axis of the COM)
        com_vel = self.data.qvel[0]  # or interpret from relevant joint velocities
        forward_reward = self.forward_reward_weight * com_vel

        # 2. Control cost (penalize large actions) calculated as -_ctrl_cost_weight * sum(action^2)
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(action))

        # 3. Stability cost (new)
        orientation = self.data.qpos[3:7]  # assuming quaternion orientation
        upright_deviation = 1 - orientation[3]  # deviation from upright pose
        stability_cost = self.stability_cost_weight * upright_deviation

        # 4. Progress bonus (new)
        if len(self.metrics_history["forward_reward"]) > 0:
            prev_x = self.metrics_history["distance_from_origin"][-1]
            curr_x = np.sqrt(self.data.qpos[0]**2 + self.data.qpos[1]**2)
            progress = curr_x - prev_x
            progress_bonus = self.progress_bonus * (progress > 0.01)  # Binary bonus for progress
        else:
            progress_bonus = 0

        # 5. Survival bonus (unchanged)
        alive_bonus = self.healthy_reward

        # Combine them into a total reward
        total_reward = (forward_reward + alive_bonus + progress_bonus - 
                       ctrl_cost - stability_cost)
        done = False  # for demonstration, we won't terminate here
        info = {}

        # Store metrics for logging
        self.metrics_history["forward_reward"].append(forward_reward)
        self.metrics_history["reward_quadctrl"].append(-ctrl_cost)  # (negative indicates penalty)
        self.metrics_history["reward_alive"].append(alive_bonus)
        self.metrics_history["distance_from_origin"].append(
            np.sqrt(self.data.qpos[0]**2 + self.data.qpos[1]**2))
        self.metrics_history["total_reward"].append(total_reward)

        return obs, total_reward, done, info

    def _get_obs(self):
        """Return the concatenated position & velocity as observation."""
        return np.concatenate([self.data.qpos, self.data.qvel])

    def compute_reward(self):
        # Existing reward components...
        
        # Add stability bonus
        up_vector = self.data.body_xpos[1] - self.data.body_xpos[0]
        up_vector = up_vector / np.linalg.norm(up_vector)
        stability_bonus = np.dot(up_vector, [0, 0, 1])  # Reward staying upright
        
        # Add smooth motion penalty
        action_smoothness = -0.1 * np.square(self.data.ctrl - self.data.ctrl).mean()
        
        # Add progress reward
        forward_velocity = self.data.qvel[0]  # Forward velocity
        progress_reward = 1.0 * forward_velocity
        
        # Combine rewards
        reward = (
            progress_reward +
            0.5 * stability_bonus +
            action_smoothness
        )
        
        # Store current action for next step
        self.last_action = self.data.ctrl.copy()
        
        # Update history
        self.reward_history.append(reward)
        
        # Normalize reward
        if len(self.reward_history) > 1:
            mean_reward = np.mean(self.reward_history)
            std_reward = np.std(self.reward_history) + 1e-8
            normalized_reward = (reward - mean_reward) / std_reward
            return normalized_reward
        
        return reward
