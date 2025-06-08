import gym
import gym.envs
import gym.envs.mujoco
import gym.envs.mujoco.mujoco_env
import gym.spaces
import numpy as np

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