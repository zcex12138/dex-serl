from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer, WindowViewer, OffScreenViewer
from typing import Optional, Dict, Literal
import mujoco
import numpy as np

class OSViewer(OffScreenViewer):
    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        img_obs_width: Optional[int] = 128,
        img_obs_height: Optional[int] = 128,
        max_geom: int = 1000,
        visual_options: Dict[int, bool] = {},
    ):
        super().__init__(model=model, 
                         data=data, 
                         width=img_obs_width, 
                         height=img_obs_height, 
                         max_geom=max_geom, 
                         visual_options=visual_options)

    def render_rgb_cam(
        self,
        render_mode: Optional[Literal["rgb_array", "depth_array", "rgbd_tuple"]] = None,
        camera_id: Optional[int] = None,
        segmentation: bool = False
    ):
        return super().render(render_mode=render_mode,camera_id=camera_id,segmentation=segmentation)


class Viewer(WindowViewer):
    def __init__(
        self,
        model: "mujoco.MjModel",
        data: "mujoco.MjData",
        width: Optional[int] = 1280,
        height: Optional[int] = 960,
        img_obs_width: Optional[int] = 128,
        img_obs_height: Optional[int] = 128,
        max_geom: int = 1000,
        visual_options: Dict[int, bool] = {},
    ):
        super().__init__(
            model=model,
            data=data,
            width=width,
            height=height,
            max_geom=max_geom,
            visual_options=visual_options,
        )

        self.img_obs_width = img_obs_width
        self.img_obs_height = img_obs_height

    def render_rgb_cam(
            self, render_mode: Optional[str],
            camera_id: Optional[int] = None,
            segmentation: bool = False
    ):
        original_cam_id = self.cam.fixedcamid
        original_type = self.cam.type
        self.cam.fixedcamid = camera_id
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.vopt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        self.viewport.width = self.img_obs_width
        self.viewport.height = self.img_obs_height
        mujoco.mjr_render(self.viewport, self.scn, self.con)

        if segmentation:
            self.scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = 1
            self.scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = 1

        rgb_arr = np.zeros(
            3 * self.viewport.width * self.viewport.height, dtype=np.uint8
        )
        depth_arr = np.zeros(
            self.viewport.width * self.viewport.height, dtype=np.float32
        )

        mujoco.mjr_readPixels(rgb_arr, depth_arr, self.viewport, self.con)

        # Process rendered images according to render_mode
        if render_mode in ["depth_array", "rgbd_tuple"]:
            depth_img = depth_arr.reshape((self.viewport.height, self.viewport.width))
            # original image is upside-down, so flip it
            depth_img = depth_img[::-1, :]
        if render_mode in ["rgb_array", "rgbd_tuple"]:
            rgb_img = rgb_arr.reshape((self.viewport.height, self.viewport.width, 3))
            # original image is upside-down, so flip it
            rgb_img = rgb_img[::-1, :]

            if segmentation:
                seg_img = (
                    rgb_img[:, :, 0]
                    + rgb_img[:, :, 1] * (2**8)
                    + rgb_img[:, :, 2] * (2**16)
                )
                seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
                seg_ids = np.full(
                    (self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32
                )

                for i in range(self.scn.ngeom):
                    geom = self.scn.geoms[i]
                    if geom.segid != -1:
                        seg_ids[geom.segid + 1, 0] = geom.objtype
                        seg_ids[geom.segid + 1, 1] = geom.objid
                rgb_img = seg_ids[seg_img]

        self.cam.fixedcamid = original_cam_id
        self.cam.type = original_type
        # Return processed images based on render_mode
        if render_mode == "rgb_array":
            return rgb_img
        elif render_mode == "depth_array":
            return depth_img
        else:  # "rgbd_tuple"
            return rgb_img, depth_img
    
    def render(self):
        super().render()