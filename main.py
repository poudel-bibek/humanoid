import mujoco
import numpy as np
import time
from environment import SimpleEnv

PATH_TO_MODEL = './mujoco_menagerie/unitree_g1/scene_with_hands.xml'

if __name__ == "__main__":
    # Create one model/data
    model = mujoco.MjModel.from_xml_path(PATH_TO_MODEL)
    data = mujoco.MjData(model)

    # Pass them into the environment
    env = SimpleEnv(model, data)

    # Launch the viewer on the exact same data
    with mujoco.viewer.launch_passive(model, data) as viewer:
        num_episodes = 3
        max_steps = 100
        
        for episode in range(num_episodes):
            obs = env.reset()
            total_reward = 0.0

            for step in range(max_steps):
                # Now data.time is the same as env.data.time
                action = 2.5 * np.sin(data.time * 2)
                
                obs, reward, done = env.step(action)
                total_reward += reward

                if done:
                    break

                # The viewer sees the exact same data
                viewer.sync()
                time.sleep(0.01)
            
            print(f"Episode {episode+1}/{num_episodes} - Total Reward: {total_reward}")
