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
    --checkpoint_path /home/jzq/github/dex-serl/examples/async_sac_state_sim_dphand/checkpoints/latest
    # --debug # wandb is disabled when debug
