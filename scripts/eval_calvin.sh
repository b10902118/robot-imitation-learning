# Examples:
# bash scripts/eval_calvin.sh CHECKPOINT_DIR

checkpoint_dir=${1}

torchrun --nproc_per_node 1 eval_calvin.py \
    --calvin_gripper_loc_bounds ./data/calvin/packaged_D_D/statistics.yaml \
    --save_video 1 \
    --checkpoint_dir ${checkpoint_dir}




                                