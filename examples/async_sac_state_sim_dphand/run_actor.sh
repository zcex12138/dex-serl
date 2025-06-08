export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_state_sim.py "$@" \
    --actor \
    --render \
    --env DphandPickCube-v0 \
    --seed 0 \
    --random_steps 1000 \
    # --debug
