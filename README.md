# humanoid

To Setup: 
Mujoco version 3.2.6 (Dec 2, 2024 Release)

Install mujoco: 

Apple Silicon: 


set the appropriate graphics backend before running: 
# export MUJOCO_GL=glfw 
# glfw for Apple Silicon
# egl for Nvidia GPU

To run: 
```
mjpython main.py
```


MJX runs on a all platforms supported by JAX: Nvidia and AMD GPUs, Apple Silicon, and Google Cloud TPUs. (https://mujoco.readthedocs.io/en/stable/mjx.html)

For GPU: 

mujoco_mjx

For MacOS, the passive viewer requires using the mjpython launcher instead of regular python