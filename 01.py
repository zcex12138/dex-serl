

import mujoco
import matplotlib.pyplot as plt
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import queue
import threading
from collections import OrderedDict
import gymnasium as gym


xml_path = '/home/jzq/github/dex-serl/franka_sim/franka_sim/envs/xmls/dpfrankacube/dphand_franka_arena.xml'
# xml_path = '/home/jzq/github/dex-serl/franka_sim/franka_sim/envs/xmls/dpfrankacube/DPhand/dphand_arena.xml'

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

view = MujocoRenderer(model,data, 128,128)
# q = view.render(render_mode="human")

cams = [model.camera(i).name for i in range(model.ncam)]

class ImageDisplayer():
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)

        self.renderer = MujocoRenderer(self.model, self.data, height=128, width=128)

        self.cams = [self.model.camera(i).name for i in range(model.ncam)]
        self.cap = OrderedDict()
        for cam in self.cams:
            self.cap[cam] = queue.Queue(maxsize=1)

        self.t = threading.Thread(target=self.get_im)
        self.t.start()

    def get_im(self):
        for idx, cam in enumerate(self.cams):
            # self.renderer.update_scene(self.data, camera=idx)
            rgb_arrays=self.renderer.render(render_mode='rgb_array')
            print(self.cap)
            print(rgb_arrays.shape)
            self.cap[cam].put(rgb_arrays)


imagedisplayer = ImageDisplayer(model=model)


import numpy as np

[-1]*7 + [-np.inf] * 22
print(dir(model))

