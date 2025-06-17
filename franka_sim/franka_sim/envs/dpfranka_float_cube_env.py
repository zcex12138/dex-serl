from pathlib import Path
from typing import Any, Literal, Tuple, Dict

import gymnasium as gym
import mujoco
import numpy as np
from jax import numpy as jnp
from gym import spaces
from franka_sim.envs.utils import *
import yaml
from mujoco import mjx
from franka_sim.controllers import opspace
from franka_sim.mujoco_gym_env import MujocoGymEnv, GymRenderingSpec
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

import queue
import threading
from collections import OrderedDict
import cv2

_HERE = Path(__file__).parent
_XML_PATH = _HERE / "xmls" / "dpfrankacube" /"dphand_franka_arena.xml"

class ImageDisplayer(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.daemon = True

    def run(self):
        while True:
            img_dict = self.queue.get()
            if img_dict is None:
                break
            frame = np.concatenate(
                [rgb for k, rgb in img_dict.items()], axis=1
            )
            cv2.imshow("cameras", frame)
            cv2.waitKey(1)




class DphandFrankaFloatCubeEnv(MujocoGymEnv):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 33}

    def __init__(
            self, 
            config_path: str,
            render_spec: GymRenderingSpec = GymRenderingSpec(),
            IMAGES: bool = False,
            ):
        super().__init__(
            xml_path=_XML_PATH,
            )
        
        self.block_id = self.data.body('block').id
        self.target_id = self.data.body('target').id
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.IMAGES = IMAGES
        self.render_spec = render_spec

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
                "block_rot": spaces.Box(
                    -np.inf, np.inf, shape=(4,), dtype=np.float32
                )
            })
            })
        if self.IMAGES:
            image_dict = {}
            for cam_id in range(self.model.ncam):
                cam = self.model.camera(cam_id)
                image_dict[cam.name] = spaces.Box(
                        low=0,
                        high=255,
                        shape=(render_spec.height, render_spec.width, 3),
                        dtype=np.uint8,
                    )
            self.observation_space["images"] = spaces.Dict(image_dict)
            self.init_cameras(render_spec=render_spec)
            self._get_im()
            self.imagedisplayer = ImageDisplayer(self.cap)
            self.imagedisplayer.start()
        else:
            # 不加相机输入，那就自由视角窗口
            self.window = MujocoRenderer(
                self.model, 
                self.data, 
                width=1280,
                height=960,
                )
        
        # 动作空间包括机械臂末端 + dphand（位置控制）
        self.action_space = spaces.Dict({
                "tcp_pos": spaces.Box(
                    low=np.asarray([-1]*7),
                    high=np.asarray([1]*7),
                    dtype = np.float32,
                ),
                'dphand_pos': spaces.Box(
                    low=np.asarray([-np.inf]*22),
                    high=np.asarray([np.inf]*22),
                    dtype = np.float32,
                )
        })

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
        # 从tcp pos，计算7维度的关节力指令：

        # tau = opspace(
        #     model=self._model,
        #     data=self._data,
        #     site_id=self._pinch_site_id,
        #     dof_ids=self._panda_dof_ids,
        #     pos=self._data.mocap_pos[0],
        #     ori=self._data.mocap_quat[0],
        #     # joint=_PANDA_HOME,
        #     gravity_comp=True,
        # )
        # self._data.ctrl[self._panda_ctrl_ids] = tau
        next_obs = {}
        self.data.ctrl += action
        mujoco.mj_step(self.model, self.data)

        # 计算observation。
        if self.IMAGES:
            self._get_im()  # 获取image以及可视化
        else:
            self.window.render(render_mode="human")  # 可视化

        # 计算reward
        rew = self._compute_reward(0.01, 0.2)
        if rew != 0:
            terminated = True
            self.reset()
            print("reward is 1, reset")
        else:
            terminated = False

        # 若逃逸则重置
        if np.max(np.abs(self.data.body('block').xpos - self.block_spawn_pos)) >= 1.25:
            self.reset()
            print("reset, block escaped")

        return next_obs, rew, terminated, False, {}
    

    
    def _compute_observation(self) -> dict:
        obs = {}
        
        # proprioception
        obs["state"] = {}
        tcp_pos = self._data.sensor("tcp_pos").data
        obs["state"]["tcp_pos"] = tcp_pos.astype(np.float32)
        tcp_quat = self._data.sensor("tcp_quat").data
        obs["state"]["tcp_quat"] = tcp_quat.astype(np.float32)
        tcp_linvel = self._data.sensor("tcp_linvel").data
        obs["state"]["tcp_linvel"] = tcp_linvel.astype(np.float32)
        tcp_angvel = self._data.sensor("tcp_angvel").data
        obs["state"]["tcp_angvel"] = tcp_angvel.astype(np.float32)

        # proprioception <- hand
        dphand_pos = []
        for sensor_id in range(self.model.nsensor):
            sensor_name = self.model.sensor(sensor_id).name
            if "dphand-" in sensor_name:
                dphand_this_pos = self._data.sensor(sensor_name).data
                dphand_pos.append(dphand_this_pos)
        obs['state']['dphand_pos'] = np.concatenate(dphand_pos, axis=0).astype(np.float32)

        # block
        obs['block_pos'] = self.block_pos.astype(np.float32)
        obs['block_quat'] = self.block_quat.astype(np.float32)

        # image 
        if self.IMAGES:
            obs["images"] = {}
            images = {}
            for cam_id in range(self.model.ncam):
                cam = self.model.camera(cam_id)
                images = self._get_im()
                obs["images"][cam.name] = images[cam.name]

        return obs


    
    def _compute_reward(self, pos_threshold: float, quat_threshold: float):

        # 位置是否接近
        self.block_pos = self.data.sensor('block_pos').data
        self.target_pos = self.data.sensor('target_pos').data
        POS_NEAR = np.max(np.abs(self.block_pos - self.target_pos)) <= pos_threshold

        # 姿态是否相似
        self.block_quat = self.data.sensor('block_quat').data
        self.target_quat = self.data.sensor('target_quat').data
        distance = self.block_quat - self.target_quat
        QUAT_NEAR = np.linalg.norm(distance) <= quat_threshold

        if POS_NEAR and QUAT_NEAR:
            rew = 1
        else:
            rew = 0
        return rew


    def reset(self):
        self.data.ctrl = self.model.key_ctrl
        self.data.qpos[-4:] = self.model.key_qpos[0][-4:]
        self.data.qpos[-7:-4] = np.random.uniform(low=[0.1, -0.4, 0], high=[0.5, 0.4, 0.0001])
        self.data.qpos[:-7] = self.model.key_qpos[0][:-7]

        self.block_spawn_pos = self.data.qpos[-7:-4].copy()
        mujoco.mj_forward(self.model, self.data)

    def init_cameras(self, render_spec):
        self._viewers = {}
        self.cap = queue.Queue()

        # 这里需要给每一个camera创建一个MujocoRenderer()对象,准备好队列
        for cam_id in range(self.model.ncam):
            cam = self.model.camera(cam_id)
            self._viewers[cam.name] = MujocoRenderer(
                self.model, 
                self.data, 
                camera_id=cam_id, 
                height=render_spec.height, 
                width=render_spec.width
                )
            
    def _get_im(self):
        rgb_cache = {}
        for cam_name, m_renderer in self._viewers.items():
            rgb_data = m_renderer.render(render_mode="rgb_array")
            rgb_cache[cam_name] = rgb_data
        self.cap.put(rgb_cache)
        return rgb_cache
    
        



if __name__ == "__main__":
    import time
    __file__ = "/home/jzq/github/dex-serl/franka_sim/franka_sim/envs/dpfranka_float_cube_env.py"
    cfg_path = '/home/jzq/github/dex-serl/franka_sim/franka_sim/envs/configs/dphand_pick_cube_env_cfg.yaml'
    _HERE = Path(__file__).parent
    _XML_PATH = _HERE / "xmls" / "dpfrankacube" /"dphand_franka_arena.xml"
    env = DphandFrankaFloatCubeEnv(
        config_path=cfg_path, 
        IMAGES=False, 
        render_spec=GymRenderingSpec(height=800, width=512)
        )
    env.reset()
    # i = 0

    while True:
        env.step(
            np.concatenate([
                np.zeros(7,), 
                np.random.uniform(-1,1,22)
                ])
            )
        
        # if i >= 100:
        #     print(env._compute_observation())
        # i += 1