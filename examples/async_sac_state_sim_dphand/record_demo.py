import gym
from tqdm import tqdm
import numpy as np
import copy
import pickle as pkl
import datetime
import os

import franka_sim

from serl_launcher.wrappers.dphand_wrappers import Fix6DPoseWrapper,TeleopIntervention

from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper
from serl_launcher.wrappers.chunking import ChunkingWrapper

if __name__ == "__main__":
    env = gym.make("DphandPickCube-v0")
    env = gym.wrappers.FlattenObservation(env)
    env = TeleopIntervention(env, ip="192.168.3.45", test=True)
    env = Fix6DPoseWrapper(env, pose=[0, 0, 0.3, -1.5707, 1.5707, 0])

    obs, _ = env.reset()

    transitions = []
    success_count = 0
    success_needed = 20
    total_count = 0
    pbar = tqdm(total=success_needed)

    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"peg_insert_{success_needed}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)

    if not os.path.exists(file_dir):
        os.mkdir(file_dir)
    if os.path.exists(file_path):
        raise FileExistsError(f"{file_name} already exists in {file_dir}")
    if not os.access(file_dir, os.W_OK):
        raise PermissionError(f"No permission to write to {file_dir}")

    while success_count < success_needed:
        actions = np.zeros((6,))
        next_obs, rew, done, truncated, info = env.step(action=actions)
        if "intervene_action" in info:
            actions = info["intervene_action"]

        transition = copy.deepcopy(
            dict(
                observations=obs,
                actions=actions,
                next_observations=next_obs,
                rewards=rew,
                masks=1.0 - done,
                dones=done,
            )
        )
        transitions.append(transition)

        obs = next_obs

        if done:
            success_count += rew
            total_count += 1
            print(
                f"{rew}\tGot {success_count} successes of {total_count} trials. {success_needed} successes needed."
            )
            pbar.update(rew)
            obs, _ = env.reset()

    with open(file_path, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {success_needed} demos to {file_path}")

    env.close()
    pbar.close()
