from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

from gym.envs.registration import register
import pathlib

CUR_PATH = pathlib.Path(__file__).parent

register(
    id="PandaPickCube-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
)
register(
    id="PandaPickCubeVision-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
register(
    id="DphandPickCube-v0",
    entry_point="franka_sim.envs:DphandPickCubeGymEnv",
    kwargs={"config_path": CUR_PATH / "./envs/configs/dphand_pick_cube_env_cfg.yaml"},
    # kwargs={"config_path": "./configs/dphand_pick_cube_env_cfg.yaml"},
)