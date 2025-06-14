from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gym
import mujoco
import numpy as np
from jax import numpy as jnp
from gym import spaces
from franka_sim.envs.utils import *
import yaml

try:
    import mujoco_py
except ImportError as e:
    MUJOCO_PY_IMPORT_ERROR = e
else:
    MUJOCO_PY_IMPORT_ERROR = None

from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv_v2

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "dphand" /"dphand_arena.xml"

class DphandPickCubeGymEnv(MujocoGymEnv_v2):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(
        self,
        config_path: str,
        seed: int = 0,
        control_dt: float = 0.02, # n-substeps = control_dt / physics_dt
        physics_dt: float = 0.002, # dt
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
    ):
        # config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        super().__init__(
            xml_path=_XML_PATH,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            time_limit=self.cfg["reset"]["time_limit"],
            render_spec=render_spec,
        )

        self.render_mode = render_mode
        self.camera_id = (0, 1)
        self.image_obs = image_obs

        # Caching.
        self._dphand_dof_ids = np.arange(28)
        self._dphand_ctrl_ids = np.arange(28)

        self._block_z = self.model.geom("block").size[2]

        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "dphand/joint_pos": spaces.Box(
                            -np.inf, np.inf, shape=(28,), dtype=np.float32
                        ),
                        "dphand/joint_vel": spaces.Box(
                            -np.inf, np.inf, shape=(28,), dtype=np.float32
                        ),
                        # "dphand/joint_torque": specs.Array(shape=(21,), dtype=np.float32),
                        "block_pos": spaces.Box(
                            -np.inf, np.inf, shape=(3,), dtype=np.float32
                        ),
                        "block_rot": spaces.Box(
                            -np.inf, np.inf, shape=(4,), dtype=np.float32
                        )
                    }
                ),
            }
        )

        if self.image_obs:
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "panda/tcp_pos": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "panda/tcp_vel": spaces.Box(
                                -np.inf, np.inf, shape=(3,), dtype=np.float32
                            ),
                            "panda/gripper_pos": spaces.Box(
                                -np.inf, np.inf, shape=(1,), dtype=np.float32
                            ),
                        }
                    ),
                    "images": gym.spaces.Dict(
                        {
                            "front": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": gym.spaces.Box(
                                low=0,
                                high=255,
                                shape=(render_spec.height, render_spec.width, 3),
                                dtype=np.uint8,
                            ),
                        }
                    ),
                }
            )

        self.action_space = self._set_action_space()

        # NOTE: gymnasium is used here since MujocoRenderer is not available in gym. It
        # is possible to add a similar viewer feature with gym, but that can be a future TODO
        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self._viewer = MujocoRenderer(
            self.model,
            self.data,
            width=1280,
            height=960
            )
        self._viewer.render(self.render_mode)

    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        # Reset hand to initial position.
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        # Reset the block to initial position.
        self.data.jnt("block").qpos[:3] = [0.20, 0.0, 0.37]
        self.data.jnt("block").qpos[3:] = random_quat()
        mujoco.mj_forward(self.model, self.data)

        # random rot
        self._set_goal(
            goal_pos=np.array([0.24, 0.0, 0.35]),
            goal_rot=random_quat()
        )

        obs = self._compute_observation()
        return obs, {}
    
    def _set_goal(self, goal_pos: np.ndarray, goal_rot: np.ndarray):
        """
        Set the goal position and orientation for the block.
        Params:
            goal_pos: np.ndarray, shape (3,)
            goal_rot: np.ndarray, shape (4,)
        """
        self.goal_pos = goal_pos
        self.goal_rot = goal_rot
        # set the mocap pose
        if self.data.mocap_pos.shape[0] > 0:
            self.data.mocap_pos[0] = self.goal_pos
            self.data.mocap_quat[0] = self.goal_rot

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        take a step in the environment.
        Params:
            action: np.ndarray

        Returns:
            observation: dict[str, np.ndarray],
            reward: float,
            done: bool,
            truncated: bool,
            info: dict[str, Any]
        """
        action = np.asarray(action, dtype=np.float32) 
        action[7] *= 0.2
        action[6] *= 0.5
        # physics step
        for i in range(int(self.control_dt / self.physics_dt)):
            self.data.ctrl[self._dphand_ctrl_ids] = action
            mujoco.mj_step(self.model, self.data)
        
        # compute observation and reward
        obs = self._compute_observation()
        rew, terminated, trucated = self._get_reward(obs, np.asarray(action))
        return obs, rew, terminated, trucated, {}

    def render(self):
        rendered_frames = []
        for cam_id in self.camera_id:
            rendered_frames.append(
                self._viewer.render(render_mode="rgb_array")
            )
        return rendered_frames

    # Helper methods.

    def _compute_observation(self) -> dict:
        obs = {}
        obs["state"] = {}

        obs["state"]["dphand/joint_pos"] = self.data.qpos[self._dphand_dof_ids].astype(np.float32)

        obs["state"]["dphand/joint_vel"] = self.data.qvel[self._dphand_dof_ids].astype(np.float32)

        if self.image_obs:
            obs["images"] = {}
            obs["images"]["front"], obs["images"]["wrist"] = self.render()
        else:
            obs["state"]["block_pos"] = self.data.sensor("block_pos").data.astype(np.float32)
            obs["state"]["block_rot"] = self.data.sensor("block_quat").data.astype(np.float32)

        if self.render_mode == "human":
            self._viewer.render(self.render_mode)

        return obs

    def _get_reward(
        self,
        obs: Dict[str, np.ndarray],
        actions: np.ndarray,
    ) -> Tuple[float, bool, bool]:
        """
        Compute the reward based on the observation and action.
        """
        return self._compute_reward(
            obs,
            actions,
            self.cfg["reward"]["dist_reward_scale"],
            self.cfg["reward"]["rot_reward_scale"],
            self.cfg["reward"]["rot_eps"],
            self.cfg["reward"]["action_penalty_scale"],
            self.cfg["reward"]["reach_goal_bonus"],
            self.cfg["reward"]["success_tolerance"],
            self.cfg["reset"]["fall_dist"],
            self.cfg["reward"]["fall_penalty"]
        )

    def _compute_reward(
            self, 
            obs: Dict,
            actions: np.ndarray,
            dist_reward_scale: float,
            rot_reward_scale: float,
            rot_eps: float,
            action_penalty_scale: float,
            reach_goal_bonus: float,
            success_tolerance: float,
            fall_dist: float,
            fall_penalty: float
            ):
        goal_dist = np.linalg.norm(obs["state"]["block_pos"] - self.goal_pos)
        rot_dist = rotation_distance(obs["state"]["block_rot"], self.goal_rot)
        
        dist_rew = goal_dist * dist_reward_scale
        rot_rew = 1.0 / (rot_dist + rot_eps) * rot_reward_scale

        action_penalty = np.sum(actions**2)
        reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

        # reset conditions
        done = rot_dist <= success_tolerance
        fall_down = goal_dist >= fall_dist
        terminated = done | fall_down
        trucated = self.time_limit_exceeded()
        # Success bonus: orientation is within `success_tolerance` of goal orientation
        reward += np.where(done, reach_goal_bonus, 0.0)
        # Fall penalty: distance to the goal is larger than a threshold
        reward += np.where(fall_down, fall_penalty, 0.0)

        return reward, terminated, trucated
    
if __name__ == "__main__":
    import franka_sim
    from serl_launcher.wrappers.dphand_wrappers import Fix6DPoseWrapper, TeleopIntervention
    env = gym.make("DphandPickCube-v0", render_mode="human")
    env = gym.wrappers.FlattenObservation(env)
    env = TeleopIntervention(env, ip="192.168.3.45", test=True)
    env = Fix6DPoseWrapper(env, pose=[0, 0, 0.3, -1.5707, 1.5707, 0])
    env.reset()
    import time
    start_time = time.time()
    # env._viewer.viewer.vopt.frame = mujoco.mjtFrame.mjFRAME_SITE
    flag = True
    cnt = 0
    while True:
        cnt += 1
        if cnt % 100 == 0:
            flag = not flag
        action = flag * env.action_space.low + (not flag) * env.action_space.high
        # random action
        # action = np.random.uniform(low=env.action_space.low, 
        #                         high=env.action_space.high, 
        #                         size=env.action_space.shape)

        obs, reward, done, truncated, info = env.step(action)
        
        # reset
        if truncated:
            obs, _ = env.reset()
            print(f"Resetting environment, reward: {reward}, done: {done}, truncated: {truncated}")
        env.render()
        real_time = time.time() - start_time
        print(info["intervene_action"])
        time.sleep(max(0, env.data.time - real_time))
    env.close()
