
import time
import numpy as np
import mujoco

PATH_TO_MODEL = './mujoco_menagerie/unitree_g1/scene_with_hands.xml'

class SimpleEnv:
    def __init__(self, model, data):
        # Use the same model and data passed in from outside
        self.model = model
        self.data = data

        self.action_dim = self.model.nu
        self.observation_dim = self.model.nq + self.model.nv

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = -np.linalg.norm(action)
        done = False
        return obs, reward, done

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])