[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/vPeV_0ye)
# RPL HW2: Imitation learning of robot policies

## Disclaimer
This assignment borrows a lot of codes from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy/tree/main), [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy) and [3D Diffuser Actor](https://github.com/nickgkan/3d_diffuser_actor).

## ️ Installation

We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 
```console
> mamba env create -f envs/conda_environment.yaml
```

but you can use conda as well: 
```console
> conda env create -f envs/conda_environment.yaml
```

Activate the environment
```console
> conda activate rpl-hw2
# gym==0.21.0 has weird dependencies on the specific version of pip and wheel
# we need to pip install modules by order
> pip install setuptools==65.5.0
> pip install wheel==0.38.4
> pip install gym==0.21.0
> pip install -r envs/requirements.txt
```

### Install CALVIN benchmark

Remember to use the latest `calvin_env` module, which fixes bugs of `turn_off_led`.  See this [post](https://github.com/mees/calvin/issues/32#issuecomment-1363352121) for detail.

```console
> git clone --recurse-submodules https://github.com/mees/calvin.git
> export CALVIN_ROOT=$(pwd)/calvin
> cd calvin
> cd calvin_env; git checkout main
> cd ..
# We don't need to install calvin_models module
> head -n -2 install.sh > tmp.sh && mv tmp.sh install.sh
> ./install.sh; cd ..
```

## ️ Push-T Benchmark
### Download Training Data
Under the repo root, create data subdirectory:
```console
# `pwd`: hw2/
> mkdir data && cd data
```

Download the corresponding zip file from [https://diffusion-policy.cs.columbia.edu/data/training/](https://diffusion-policy.cs.columbia.edu/data/training/)
```console
# `pwd`: hw2/data
> wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
> unzip pusht.zip && rm -f pusht.zip && cd ..
```

### Training

```console
# bash scripts/train_policy.sh METHOD_NAME SEED_ID
# For diffusion modeling, set METHOD_NAME to train_diffusion_unet_hybrid_pusht_workspace 
# For regression modeling, set METHOD_NAME to train_regression_unet_hybrid_pusht_workspace 
> bash scripts/train_policy.sh train_diffusion_unet_hybrid_pusht_workspace 0
```

This will create a directory in format `data/outputs/yyyy-mm-dd_hh-mm-ss-<method_name>` where configs, logs and checkpoints are written to. The policy will be evaluated every 50 epochs with the success rate logged as `test/mean_score` on wandb, as well as videos for some rollouts.
```console
> tree data/outputs/YEAR-MONTH-DATE_HOUR-MINUTE-SECOND-METHOD_NAME -I wandb

data/outputs/YEAR.MONTH.DATE/HOUR.MINUTE.SECOND-METHOD_NAME
├── checkpoints
│   ├── epoch=0000-test_mean_score=0.134.ckpt
│   └── latest.ckpt
├── .hydra
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
├── logs.json.txt
├── media
│   ├── 2k5u6wli.mp4
│   ├── 2kvovxms.mp4
│   ├── 2pxd9f6b.mp4
│   ├── 2q5gjt5f.mp4
│   ├── 2sawbf6m.mp4
│   └── 538ubl79.mp4
└── train.log

3 directories, 13 files
```

###  Evaluate Trained Checkpoints

```console
> python eval.py --checkpoint data/outputs/YEAR-MONTH-DATE_HOUR-MINUTE-SECOND-METHOD_NAME/checkpoints/latest.ckpt --output_dir data/pusht_eval_output --device cuda:0
```

This will generate the following directory structure:
```console
> tree data/pusht_eval_output
data/pusht_eval_output
├── eval_log.json
└── media
    ├── seed100000.mp4
    ├── seed100001.mp4
    ├── seed100002.mp4
    └── seed100003.mp4

1 directory, 7 files
```

`eval_log.json` contains metrics that is logged to wandb during training:
```console
> cat data/pusht_eval_output/eval_log.json
{
  "test/mean_score": 0.9150393806777066,
  "test/sim_max_reward_4300000": 1.0,
  "test/sim_max_reward_4300001": 0.9872969750774386,
...
  "train/sim_video_1": "data/pusht_eval_output//media/2fo4btlf.mp4"
}
```

## ️ CALVIN Benchmark

[Calvin](https://github.com/mees/calvin) is a benchmark for table-top manipulation, where the robot is commanded by language instruction complete manipulation tasks **in a row**, using only visual observations.

The scene includes a desk, a drawer, a cabinet, a switch, a button, a lightbulb, an LED light, and three blocks of different colors.

The manipulation tasks include:
```
- push the block
- lift up the block
- rotate the block
- put the block into the cabinet/drawer
- take out the block from the cabinet/drawer
- stack/unstack the blocks
- open/close the drawer
- turn on/off the LED light
- turn on/off the lightbulb.
```

However, due to the limited computational resources and model capacity, we only consider `open_drawer` and `close_drawer` task in this assignment.  You will train the implemented diffusion policy on these two tasks **offline** (using offline demonstrations), and evaluate the trained model **online** (interact with the environment with your policy rollouts).

### Download Training Data

Download the corresponding zip file from [https://huggingface.co/datasets/twke/RPL2024-hw2/blob/main/calvin.zip](https://huggingface.co/datasets/twke/RPL2024-hw2/blob/main/calvin.zip)
```console
# `pwd`: hw2/data
> unzip calvin.zip && rm -f calvin.zip && cd ..
```

### Training

The training procedure is the same as PushT benchmark.

```console
> bash scripts/train_policy.sh train_diffusion_unet_hybrid_calvin_workspace 0
```

###  Evaluate Trained Checkpoints

```console
>  bash scripts/eval_calvin.sh data/outputs/YEAR-MONTH-DATE_HOUR-MINUTE-SECOND-METHOD_NAME
```