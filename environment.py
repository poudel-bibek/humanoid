
import time
import numpy as np
import mujoco

PATH_TO_MODEL = './mujoco_menagerie/unitree_g1/scene_with_hands.xml'

class UnitreeG1:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
        self.data = mujoco.MjData(self.model)

    def reset(self):
        pass 

    def step(self, action):
        pass 

    def render(self):
        pass 

    def close(self):
        pass 

    def get_obs(self):
        pass 

    def compute_reward(self):
        pass 
