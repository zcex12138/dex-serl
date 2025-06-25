from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from franka_sim.envs.render import Viewer, OSViewer

@dataclass(frozen=True)
class GymRenderingSpec:
    height: int = 128
    width: int = 128
    viewer_height: int = 960
    viewer_width: int = 1280
    camera_id: str | int = -1


class MujocoGymEnv(MujocoEnv):
    """MujocoEnv with gym interface."""
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 100}
    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = float("inf"),
        render_spec: GymRenderingSpec = GymRenderingSpec(),
        render_mode: Literal["rgb_array", "human"] = "rgb_array"
    ):
        self.metadata["render_fps"] = np.round(1.0 / control_dt) 

        super().__init__(xml_path.as_posix(), 
                         frame_skip=int(control_dt // physics_dt), 
                         observation_space=None, 
                         render_mode=render_mode
                        )
        
        self._control_dt = control_dt
        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        
        self._render_specs = render_spec
        self._viewer = None
        self.render_mode = render_mode

        self._init_viewer()

    def _init_viewer(self):
        if self.render_mode == "human":
            self._viewer = Viewer( # 交互式渲染
                self.model,
                self.data,
                width=self._render_specs.viewer_width,
                height=self._render_specs.viewer_height,
                img_obs_width=self._render_specs.width,
                img_obs_height=self._render_specs.height,
            )
        elif self.render_mode == "rgb_array":
            self._viewer = OSViewer( # 离屏渲染
                self.model,
                self.data,
                img_obs_width=self._render_specs.width,
                img_obs_height=self._render_specs.height,
            )


    def time_limit_exceeded(self) -> bool:
        return self.data.time >= self._time_limit

    # Accessors.
    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self.model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random
    
    def render(self):
        if self.render_mode == "human":
            self._viewer.render()
        elif self.render_mode == "rgb_array":
            return self._viewer.render_rgb_cam("rgb_array", -1)

    def close(self):
        self._viewer.close()
        self._viewer = None