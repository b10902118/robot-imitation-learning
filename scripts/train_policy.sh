# Examples:
# bash scripts/train_policy.sh MODEL_NAME SEED_ID
# bash scripts/train_policy.sh train_diffusion_unet_hybrid_pusht_workspace 0

save_ckpt=True
eval_only=False
config_name=${1}
seed=${2}
run_dir=data/outputs/$(date +%Y-%m-%d_%H-%M-%S)-${config_name}

export HYDRA_FULL_ERROR=1 
python train.py \
    --config-dir=./diffusion_policy/config/ \
    --config-name=${config_name}.yaml \
    hydra.run.dir=${run_dir} \
    training.seed=${seed} \
    training.device="cuda:0" \
    +checkpoint.save_ckpt=${save_ckpt}
