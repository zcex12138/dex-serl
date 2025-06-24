export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.5 && \
python async_sac_state_sim_float.py "$@" \
    --actor \
    --render \
    --env DphandFrankaFloatCube-v0 \
    --seed 0 \
    --random_steps 1000 \
    --debug
    # --image_obs \
