import gymnasium as gym
from pynput import keyboard
import numpy as np
from dphand_teleop.dphand_teleoperator import DPhandTeleoperator

class TeleopIntervention(gym.Wrapper):
    def __init__(self, env, ip="192.168.3.27", test=True):
        super().__init__(env)
        self.expert = DPhandTeleoperator(env.model, env.data, ip, test=test)
        self.listener.start()
        self.intervened = False
    
    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: teleop action if nonezero; else, policy action
        """
        def on_press(key):
            if key == 'Key.space':
                self.intervened = not self.intervened

        self.listener = keyboard.Listener(on_press=on_press)

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
    
    # 咱们的teleop干预没有准备close()方法吗
    def close(self):
        self.listener.stop()
    #     self.expert.close()
    #     super().close()

        
