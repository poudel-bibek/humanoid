import mujoco
import numpy as np

class SimpleEnv:
    def __init__(self, model, data, forward_reward_weight=1.0, ctrl_cost_weight=0.01, healthy_reward=1.0):
        """
        A simple environment that calculates a more complex reward.

        Args:
            model (MjModel): The MuJoCo model.
            data (MjData): The MuJoCo data associated with model.
            forward_reward_weight (float): Weight for forward velocity reward.
            ctrl_cost_weight (float): A penalty for using too much control/action
            healthy_reward (float): Constant bonus reward for 'being alive'.
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

        # For logging
        self.metrics_history = {
            "forward_reward": [],
            "reward_quadctrl": [],
            "reward_alive": [],
            "distance_from_origin": [],
            "total_reward": []
        }

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
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

        # 3. Survival bonus (always alive in this simple example)
        alive_bonus = self.healthy_reward

        # 4. Distance from origin (just an example metric — can be used as penalty or for logging)
        #    We'll measure the distance of the main body (index 0) from origin
        x_pos, y_pos = self.data.qpos[0], self.data.qpos[1]
        dist_from_origin = np.sqrt(x_pos**2 + y_pos**2)

        # Combine them into a total reward
        total_reward = forward_reward + alive_bonus - ctrl_cost
        done = False  # for demonstration, we won't terminate here
        info = {}

        # Store metrics for logging
        self.metrics_history["forward_reward"].append(forward_reward)
        self.metrics_history["reward_quadctrl"].append(-ctrl_cost)  # (negative indicates penalty)
        self.metrics_history["reward_alive"].append(alive_bonus)
        self.metrics_history["distance_from_origin"].append(dist_from_origin)
        self.metrics_history["total_reward"].append(total_reward)

        return obs, total_reward, done, info

    def _get_obs(self):
        """Return the concatenated position & velocity as observation."""
        return np.concatenate([self.data.qpos, self.data.qvel])
