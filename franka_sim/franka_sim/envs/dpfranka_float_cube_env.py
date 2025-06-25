from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np
from jax import numpy as jnp
from franka_sim.envs.utils import *
import yaml

# from mujoco import mjx
from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv


_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "dpfrankacube" /"dphand_franka_arena.xml"

class DphandFrankaFloatCubeEnv(MujocoGymEnv):
    def __init__(
            self, 
            config_path: str,
            seed: int = 0,
            action_scale: np.ndarray = np.asarray([0.1, 1]),
            render_spec: GymRenderingSpec = GymRenderingSpec(),
            render_mode: Literal["human", "rgb_array", "depth_array"] = "rgb_array",
            image_obs: bool = False,
            ):

        # config
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        super().__init__(
            xml_path=_XML_PATH,
            seed = seed,
            control_dt=0.02,
            physics_dt=0.002,
            time_limit=10.0,
            render_spec=render_spec,
            render_mode=render_mode
            )
        
        self.image_obs = image_obs

        self.block_id = self.data.body('block').id
        self.target_id = self.data.body('target').id

        # store last state
        self._obs = None
        self._rew = None
        self._action = None
        self._action_scale = action_scale

        # 观测空间包括proprioception, image*2, touch(待加入)
        self.observation_space = spaces.Dict({
            'state': spaces.Dict({
                "tcp_pos": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float32
                ),
                "tcp_quat": spaces.Box(
                    -np.inf, np.inf, shape=(4,), dtype=np.float32
                ),
                "tcp_linvel": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float32
                ),
                "tcp_angvel": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float32
                ),
                "dphand_pos": spaces.Box(
                    -np.inf, np.inf, shape=(22,), dtype=np.float32
                ),
                "block_pos": spaces.Box(
                    -np.inf, np.inf, shape=(3,), dtype=np.float32
                ),
                "block_quat": spaces.Box(
                    -np.inf, np.inf, shape=(4,), dtype=np.float32
                )
            })
            })
        
        if self.image_obs:
            self.cam_names = ['cam_front', 'cam_right', 'cam_up', 'handcam_rgba']
            image_dict = {}
            for name in self.cam_names:
                image_dict[name] = spaces.Box(
                        low=0,
                        high=255,
                        shape=(render_spec.height, render_spec.width, 3),
                        dtype=np.uint8,
                    )
            self.observation_space["images"] = spaces.Dict(image_dict)

        # Caching.
        self._panda_dof_ids = np.asarray(
            [self.model.joint(f"joint{i}").id for i in range(1, 8)]
        )
        self._panda_ctrl_ids = np.asarray(
            [self.model.actuator(f"actuator{i}").id for i in range(1, 8)]
        )
        self._pinch_site_id = self.model.site("attachment_site").id
        
        _dphand_ctrl_ids = []
        for sensor_id in range(self.model.nsensor):
            sensor_name = self.model.sensor(sensor_id).name
            if "dphand-" in sensor_name:
                ctrl_id = self.model.joint(sensor_name.removeprefix("dphand-").removesuffix("_pos")).id
                _dphand_ctrl_ids.append(ctrl_id)
        self._dphand_ctrl_ids = np.asarray(_dphand_ctrl_ids)

        # print(f"arm dof ids {self._panda_dof_ids}, arm ctrl ids {self._panda_ctrl_ids}, dphand ctrl ids {self._dphand_ctrl_ids}")
        # 动作空间包括机械臂末端 + dphand（位置控制）
        self.action_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf,
            shape=(7 + 22,),  # 3 for tcp pos,
            dtype=np.float32,
        )

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

        delta_tcp_pos = action[:3]
        delta_tcp_quat = action[3:7]
        
        tcp_pos = self.data.sensor('tcp_pos').data
        tcp_quat = self.data.sensor('tcp_quat').data

        desired_tcp_pos = tcp_pos + delta_tcp_pos * self._action_scale[0]
        desired_tcp_quat = tcp_quat + delta_tcp_quat * self._action_scale[0]

        # franka 关节指令的发布
        # 从tcp pos，计算7维度的关节力指令：
        tau = opspace(
            model=self.model,
            data=self.data,
            joint=None,
            site_id=self._pinch_site_id,
            dof_ids=self._panda_dof_ids,
            pos=desired_tcp_pos,
            ori=desired_tcp_quat,
            gravity_comp=False,
        )
        self.data.ctrl[self._panda_ctrl_ids] = tau

        # 灵巧手指令的发布，因为是pos所以简单很多
        self.data.ctrl[self._dphand_ctrl_ids] = action[7:]

        for _ in range(5):
            self.data.ctrl += action
            mujoco.mj_step(self.model, self.data)

        self._obs = self._compute_observation()  # 获取image以及可视化

        terminated = False
        # 计算reward
        self._rew = self._compute_reward(0.01, 0.2)
        if self._rew != 0:
            terminated = True
            # print("sucess, reset")

        # 若逃逸则重置
        if np.max(np.abs(self.data.body('block').xpos - self.block_spawn_pos)) >= 0.8:
            terminated = True
            # print("reset, block escaped")
        
        trucated = self.time_limit_exceeded()
        return self._obs, self._rew, terminated, trucated, {}
    

    
    def _compute_observation(self) -> dict:
        obs = {}
        # proprioception
        obs["state"] = {}
        tcp_pos = self.data.sensor("tcp_pos").data
        obs["state"]["tcp_pos"] = tcp_pos.astype(np.float32)
        tcp_quat = self.data.sensor("tcp_quat").data
        obs["state"]["tcp_quat"] = tcp_quat.astype(np.float32)
        tcp_linvel = self.data.sensor("tcp_linvel").data
        obs["state"]["tcp_linvel"] = tcp_linvel.astype(np.float32)
        tcp_angvel = self.data.sensor("tcp_angvel").data
        obs["state"]["tcp_angvel"] = tcp_angvel.astype(np.float32)

        # proprioception <- hand
        dphand_pos = []
        for sensor_id in range(self.model.nsensor):
            sensor_name = self.model.sensor(sensor_id).name
            if "dphand-" in sensor_name:
                dphand_this_pos = self.data.sensor(sensor_name).data
                dphand_pos.append(dphand_this_pos)
        
        obs['state']['dphand_pos'] = np.concatenate(dphand_pos, axis=0).astype(np.float32)

        # block
        self.block_pos = self.data.sensor('block_pos').data
        self.block_quat = self.data.sensor('block_quat').data
        obs['state']['block_pos'] = self.block_pos.astype(np.float32)
        obs['state']['block_quat'] = self.block_quat.astype(np.float32)

        # image 
        if self.image_obs:
            obs["images"] = {}
            for cam_id in range(self.model.ncam):
                image = self._viewer.render_rgb_cam('rgb_array', camera_id=cam_id)
                obs["images"][self.model.camera(cam_id).name] = image

        return obs


    
    def _compute_reward(self, pos_threshold: float, quat_threshold: float):
        # 位置是否接近
        self.block_pos = self.data.sensor('block_pos').data
        self.target_pos = self.data.sensor('target_pos').data
        pos_near = np.max(np.abs(self.block_pos - self.target_pos)) <= pos_threshold

        # 姿态是否相似
        self.block_quat = self.data.sensor('block_quat').data
        self.target_quat = self.data.sensor('target_quat').data
        distance = self.block_quat - self.target_quat
        quat_near = np.linalg.norm(distance) <= quat_threshold

        if pos_near and quat_near:
            rew = 1
        else:
            rew = 0
        return rew


    def reset(
        self, seed=None, **kwargs
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl = self.model.key_ctrl
        self.data.qpos[-4:] = self.model.key_qpos[0][-4:]
        self.data.qpos[-7:-4] = np.random.uniform(low=[0.1, -0.4, 0], high=[0.5, 0.4, 0.0001])
        self.data.qpos[:-7] = self.model.key_qpos[0][:-7]

        self.block_spawn_pos = self.data.qpos[-7:-4].copy()
        mujoco.mj_forward(self.model, self.data)

        self._obs = self._compute_observation()
    
        return self._obs, {}


if __name__ == "__main__":
    import franka_sim
    env = gym.make("DphandFrankaFloatCube-v0", render_mode="human", image_obs=True)
    env = gym.wrappers.FlattenObservation(env)
    env_unwrapped = env.unwrapped
    env.reset()

    import cv2

    while True:
        obs, _, done, trucated, _ = env.step(
                        np.concatenate([
                            np.asarray([1, 0, -1.5]),
                            np.zeros(26,), 
                            ])
                    )
        
        obs_unflatten = env_unwrapped._obs
        if env_unwrapped.image_obs:
            image = np.hstack(
                [obs_unflatten["images"][k] for k in env_unwrapped.cam_names if k in obs_unflatten["images"]]
            )
            cv2.imshow("image_obs", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        if done or trucated:
            env.reset()
        env.render()