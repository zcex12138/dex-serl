export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_state_sim.py "$@" \
    --player \
    --render \
    --env DphandPickCube-v0 \
    --exp_name=serl_dev_sim_test \
    --seed 0 \
    --random_steps 1000 \
    --checkpoint_path /home/jzq/github/dex-serl/examples/async_sac_state_sim_dphand/checkpoints/latest
    # --debug
