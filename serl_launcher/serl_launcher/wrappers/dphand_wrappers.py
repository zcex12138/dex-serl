import gym
import gym.spaces
import numpy as np
from pynput import keyboard

from serl_launcher.dphand_teleop.dphand_teleoperator import DPhandTeleoperator
import mujoco

class Fix6DPoseWrapper(gym.ActionWrapper):
    """
    A wrapper to fix the 6D pose of the dexterous hand in the environment.
    This wrapper modifies the action space to only allow changes in the gripper's
    position and orientation, while keeping the 6D pose fixed.
    """
    def __init__(self, env, pose=np.zeros(6)):
        super().__init__(env)
        # Assuming the action space is a Box with shape (n,), where n is the number of actions
        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("The action space must be a Box space.")
        self.action_space = gym.spaces.Box(
            low=env.action_space.low[6:], 
            high=env.action_space.high[6:],
            shape=(self.env.action_space.shape[0] - 6,), 
            dtype=np.float32
        )
        self.pose = np.array(pose, dtype=np.float32)

    def action(self, action):
        """
        Modify the action to fix the 6D pose.
        The first 6 elements of the action are ignored.
        """
        if len(action) < 6:
            raise ValueError("Action must have at least 6 elements.")
        # Return the action with the first 6 elements ignored
        return np.concatenate((self.pose, action))

class TeleopIntervention(gym.Wrapper):
    def __init__(self, env, ip="192.168.3.27", test=True):
        super().__init__(env)
        data_c = mujoco.MjData(env.model)
        self.expert = DPhandTeleoperator(env.model, data_c, ip, test=test, n_step=1)
        self.intervened = True
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        if key == 'Key.space':
            self.intervened = not self.intervened
    
    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: teleop action if nonezero; else, policy action
        """
        if self.intervened:
            expert_a = self.expert.get_target_action_j2j()
            expert_a[:6] = self.model.key_qpos[0, :6]  # 手腕不能乱运动
            return expert_a, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        return obs, rew, done, truncated, info
    
    def close(self):
        self.listener.stop()
        
