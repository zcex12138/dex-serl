# Dex-SERL: Sample-Efficient Reinforcement Learning for Dexterous Manipulation

[Raw Repository](https://github.com/rail-berkeley/serl) (click here)

## Installation
1. **Install the SERL**

2. **Install the [DehandTeleop](https://github.com/zcex12138/dphand-teleop)**


**Major Changes**

| Code Directory | Description |
| --- | --- |
| [dphand_pick_cube_env.py](https://github.com/rail-berkeley/serl/blob/main/serl_launcher) | Main code for SERL |
| [serl_launcher.agents](https://github.com/rail-berkeley/serl/blob/main/serl_launcher/serl_launcher/agents/) | Agent Policies (e.g. DRQ, SAC, BC) |
| [serl_launcher.wrappers](https://github.com/rail-berkeley/serl/blob/main/serl_launcher/serl_launcher/wrappers) | Gym env wrappers |
| [serl_launcher.data](https://github.com/rail-berkeley/serl/blob/main/serl_launcher/serl_launcher/data) | Replay buffer and data store |
| [serl_launcher.vision](https://github.com/rail-berkeley/serl/blob/main/serl_launcher/serl_launcher/vision) | Vision related models and utils |
| [franka_sim](./franka_sim) | Franka mujoco simulation gym environment |
| [serl_robot_infra](./serl_robot_infra/) | Robot infra for running with real robots |
| [serl_robot_infra.robot_servers](https://github.com/rail-berkeley/serl/blob/main/serl_robot_infra/robot_servers/) | Flask server for sending commands to robot via ROS |
| [serl_robot_infra.franka_env](https://github.com/rail-berkeley/serl/blob/main/serl_robot_infra/franka_env/) | Gym env for real franka robot |