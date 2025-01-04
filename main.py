import mujoco
import mujoco.viewer
import numpy as np

PATH_TO_MODEL = './mujoco_menagerie/unitree_g1/scene_with_hands.xml'
model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)

# # Change the step size
# for _ in range(2000):

#     data.ctrl = 1.5 * np.random.randn(model.nu)
#     mujoco.mj_step(model, data)
#     viewer.sync()

# To run forever
while True: 
    viewer.sync()