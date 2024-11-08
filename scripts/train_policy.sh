# Examples:
# bash scripts/train_policy.sh MODEL_NAME SEED_ID
# bash scripts/train_policy.sh train_diffusion_unet_hybrid_pusht_workspace 0

save_ckpt=True
eval_only=False
seed=0

type=${1}
case ${type} in
    "diff_calvin")
        config_name="train_diffusion_unet_hybrid_calvin_workspace"
        run_dir="diff_calvin"
        ;;
    "diff_pusht")
        config_name="train_diffusion_unet_hybrid_pusht_workspace"
        run_dir="diff_pusht"
        ;;
    "reg_pusht")
        config_name="train_regression_unet_hybrid_pusht_workspace"
        run_dir="reg_pusht"
        ;;
    *)
        echo "Invalid type specified"
        exit 1
        ;;
esac
#config_name=${1}
run_dir=data/outputs/${run_dir}

export HYDRA_FULL_ERROR=1 
python train.py \
    --config-dir=./diffusion_policy/config/ \
    --config-name=${config_name}.yaml \
    hydra.run.dir=${run_dir} \
    training.seed=${seed} \
    training.device="cuda:0" \
    +checkpoint.save_ckpt=${save_ckpt}
