
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
import mujoco

xml_path = "/home/jzq/github/dex-serl/franka_sim/franka_sim/envs/xmls/dpfrankacube/dphand_franka_arena.xml"


model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

viewer0 = MujocoRenderer(model, data, camera_id=0, height=1280, width=1280)
viewer1 = MujocoRenderer(model, data, camera_id=1, height=512, width=512)
viewer2 = MujocoRenderer(model, data, camera_id=2, height=1280, width=1280)

while True:
    mujoco.mj_step(model, data)
    viewer1.render(render_mode="human")
    print(viewer1)

    # mujoco.mj_step(model, data)
    # r1 = viewer1.render(render_mode="rgb_array")
    # print(r1)