import gym
import numpy as np
import copy
import pickle as pkl
import datetime
import os

import franka_sim
from franka_sim.envs.dphand_pick_cube_env import DphandPickCubeGymEnv

from serl_launcher.wrappers.dphand_wrappers import Fix6DPoseWrapper,TeleopIntervention

from pynput import keyboard

def on_press(key):
    if key == keyboard.KeyCode.from_char('p'):
        intervened = not intervened

listener = keyboard.Listener(on_press=on_press)
listener.start()

if __name__ == "__main__":
    env = gym.make("DphandPickCube-v0", render_mode="human")
    env_unwrapped : DphandPickCubeGymEnv = env.unwrapped
    env = gym.wrappers.FlattenObservation(env)
    env = TeleopIntervention(env, ip="192.168.3.8", test=False)
    env = Fix6DPoseWrapper(env, pose=[0, 0, 0.3, -1.5707, 1.5707, 0])

    obs, _ = env.reset()

    transitions = []
    transitions_episode = []
    success_count = 0
    total_count = 0


    start = False
    while env.keyboard != "esc":
        if start == False and env.keyboard == "enter":
            obs, _ = env.reset()
            obs_unflatten = env_unwrapped._obs
            start = True
            print("Start recording demos.\n Please press 'r' to reset recording or 'enter' to finish recording.\n")
        elif start == True:
            actions = info["action"]
            replace = info["replaced"]
            transition = copy.deepcopy(
                dict(
                    observations=obs,
                    observations_unflatten=obs_unflatten,
                    actions=actions,
                    replaced=replace,
                    next_observations=next_obs,
                    rewards=rew,
                    masks=1.0 - done,
                    dones=done,
                )
            )
            transitions_episode.append(transition)
            obs = next_obs
            obs_unflatten = next_obs_unflatten

            if env.keyboard == "enter":
                success_count += info['success']
                total_count += 1
                print(f"Got total {total_count} demos.\nPlease press 'enter' to continue or 'esc' to save demos.\n")

                # 重新计算reward
                env_unwrapped.goal_rot = transition["observations_unflatten"]["state"]["block_rot"]
                for transition in transitions_episode:
                    transition["rewards"] = env_unwrapped._get_reward(transition["observations_unflatten"], 
                                                    np.concatenate(([0, 0, 0.3, -1.5707, 1.5707, 0], transition["actions"])))
                    del transition["observations_unflatten"]
                
                transitions += transitions_episode
                transitions_episode = []

                obs, _ = env.reset()
                obs_unflatten = env_unwrapped._obs
                start = False
            
            if env.keyboard == "r":
                print("Re-start recording demos.\n")
                obs, _ = env.reset()
                obs_unflatten = env_unwrapped._obs
                transitions_episode = []
                continue
                

        actions = np.zeros((6,))
        next_obs, rew, done, truncated, info = env.step(action=actions)
        next_obs_unflatten = env_unwrapped._obs

    # Save
    uuid = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"dphand_{total_count}_demos_{uuid}.pkl"
    file_dir = os.path.dirname(os.path.realpath(__file__))  # same dir as this script
    file_path = os.path.join(file_dir, file_name)
    with open(file_path, "wb") as f:
        pkl.dump(transitions, f)
        print(f"saved {total_count} demos to {file_path}")

    env.close()
