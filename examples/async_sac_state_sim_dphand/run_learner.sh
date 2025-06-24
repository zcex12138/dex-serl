export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_state_sim.py "$@" \
    --learner \
    --env DphandPickCube-v0 \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --training_starts 1000 \
    --critic_actor_ratio 8 \
    --batch_size 64 \
    --checkpoint_period 50000 \
    --checkpoint_path /home/yhx/workspace/serl/examples/async_sac_state_sim_dphand/checkpoints/ \
    --demo_path /home/yhx/workspace/serl/examples/async_sac_state_sim_dphand/dphand_20_demos_2025-06-18_17-12-34.pkl \
    --debug # wandb is disabled when debug
