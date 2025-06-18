

import franka_sim
from agentlace.trainer import TrainerServer, TrainerClient
from agentlace.data.data_store import QueuedDataStore

from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.wrappers.serl_obs_wrappers import SERLObsWrapper

from serl_launcher.utils.launcher import (
    make_sac_agent,
    make_trainer_config,
    make_wandb_logger,
    make_replay_buffer,
)
from absl import flags, app

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "HalfCheetah-v4", "Name of environment.")
flags.DEFINE_string("agent", "sac", "Name of agent.")
flags.DEFINE_string("exp_name", None, "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 8, "critic to actor update ratio.")
flags.DEFINE_integer("max_steps", 500000, "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 500000, "Replay buffer capacity.")
flags.DEFINE_integer("random_steps", 300, "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 300, "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30, "Number of steps per update the server.")
flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")   # flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("player", False, "Create a player.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_integer("checkpoint_period", 0, "Period to save checkpoints.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_integer("utd_ratio", 1, "UTD ratio for SAC.")
flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")
flags.DEFINE_boolean("debug", False, "Debug mode.")  # debug mode will disable wandb logging
flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")
# flags.DEFINE_integer("eval_n_trajs", 5, "Number of trajectories for evaluation.")


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


##############################################################################


def actor(agent: SACAgent, env, data_store):

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    for step in range(FLAGS.max_steps):
        o

def learner(agent: SACAgent, databuffer):
    pass






def main():
    from franka_sim.envs.dpfranka_float_cube_env import DphandFrankaFloatCubeEnv
    __file__ = "/home/jzq/github/dex-serl/franka_sim/franka_sim/envs/dpfranka_float_cube_env.py"
    cfg_path = '/home/jzq/github/dex-serl/franka_sim/franka_sim/envs/configs/dphand_pick_cube_env_cfg.yaml'
    _HERE = Path(__file__).parent
    _XML_PATH = _HERE / "xmls" / "dpfrankacube" /"dphand_franka_arena.xml"
    env = DphandFrankaFloatCubeEnv(
        config_path=cfg_path, 
        IMAGES=False, 
        )
    env.reset()
    # i = 0

    while True:
        q = env.step(
            np.concatenate([
                np.asarray([1, 0, -1.5]),
                np.zeros(26,), 
                ])
            )
    pass


if __name__ == "__main__":
    app.run(main)