# import mujoco
# import mediapy as media

# model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
# data = mujoco.MjData(model)

# print(model.ngeom)
# print(model.geom_rgba)


# with mujoco.Renderer(model) as renderer:
#   mujoco.mj_forward(model, data)
#   renderer.update_scene(data)

#   media.show_image(renderer.render())



        
        # Optional: add small delay to make motion more visible
        #time.sleep(0.01)




import mujoco
import mujoco.viewer
import numpy as np

PATH_TO_MODEL = './mujoco_menagerie/unitree_g1/scene_with_hands.xml'
model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
data = mujoco.MjData(model)

viewer = mujoco.viewer.launch_passive(model, data)

control_limits= np.array([
    [0, 0], # hip 
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0],
])

control_ranges = control_limits[:, 1] - control_limits[:, 0]  # (max - min) for each control
control_mins = control_limits[:, 0]  # min values for each control

# # Change the step size
# for _ in range(2000):

#     data.ctrl = 1.5 * np.random.randn(model.nu)
#     mujoco.mj_step(model, data)
#     viewer.sync()

# To run forever
while True: 
    viewer.sync()