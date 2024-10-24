"""Modified from
https://github.com/mees/calvin/blob/main/calvin_models/calvin_agent/evaluation/evaluate_policy.py
"""
if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
import os
import gc
from typing import Tuple, Optional, List
import random
import logging
from pathlib import Path

import tap
import hydra
from omegaconf import OmegaConf
import torch
import numpy as np
import yaml
from tqdm import tqdm

from calvin_utils.common_utils import get_gripper_loc_bounds
from calvin_utils.online_evaluation_calvin.evaluate_model import create_model
from calvin_utils.online_evaluation_calvin.evaluate_utils import (
    prepare_visual_states,
    prepare_proprio_states,
    count_success,
    get_env_state_for_initial_condition,
    collect_results,
    write_results,
    get_log_dir
)
from calvin_utils.online_evaluation_calvin.singlestep_sequences import get_sequences
from calvin_utils.online_evaluation_calvin.evaluate_utils import get_env

logger = logging.getLogger(__name__)

EP_LEN = 360  # 120 would be better
NUM_SEQUENCES = 100


class Arguments(tap.Tap):
    # Online enviornment
    calvin_dataset_path: Path = "./data/calvin/packaged_D_D"
    calvin_model_path: Path = "./calvin/calvin_models"
    device: str = "cuda"
    save_video: int = 0

    # Offline data loader
    seed: int = 0
    checkpoint_dir: Path
    calvin_gripper_loc_bounds: Optional[str] = None
    config_name: str = "train_diffusion_unet_hybrid_calvin_workspace"

    # Logging to base_log_dir/exp_log_dir/run_log_dir
    base_log_dir: Path = Path(__file__).parent / "data" / "calvin_eval_output"
    relative_action: int = 1


def make_env(dataset_path, show_gui=True, scene=None):
    val_folder = Path(dataset_path)
    if scene is not None:
        env = get_env(val_folder, show_gui=show_gui, scene=scene)
    else:
        env = get_env(val_folder, show_gui=show_gui)

    return env


def evaluate_policy(model, env, conf_dir, eval_log_dir=None, save_video=False,
                    sequence_indices=[]):
    """
    Run this function to evaluate a model on the CALVIN challenge.

    Args:
        model: an instance of CalvinBaseModel
        env: an instance of CALVIN_ENV
        conf_dir: Path to the directory containing the config files of CALVIN
        eval_log_dir: Path where to log evaluation results
        save_video: a boolean indicates whether to save the video
        sequence_indices: a list of integers indicates the indices of the
            instruction chains to evaluate

    Returns:
        results: a list of integers indicates the number of tasks completed
    """
    task_cfg = OmegaConf.load(conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml")
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(conf_dir / "annotations/new_playtable_validation.yaml")

    eval_log_dir = get_log_dir(eval_log_dir)

    eval_sequences = get_sequences(NUM_SEQUENCES)

    results, tested_sequence_indices = collect_results(eval_log_dir)

    for seq_ind, (initial_state, eval_sequence) in enumerate(eval_sequences):
        if sequence_indices and seq_ind not in sequence_indices:
            continue
        if seq_ind in tested_sequence_indices:
            continue
        result, videos = evaluate_sequence(
            env, model, task_oracle, initial_state,
            eval_sequence, val_annotations, save_video
        )
        write_results(eval_log_dir, seq_ind, result)
        results.append(result)
        str_results = (
            " ".join([f"{i + 1}/5 : {v * 100:.1f}% |"
            for i, v in enumerate(count_success(results))]) + "|"
        )
        print(str_results + "\n")

        if save_video:
            import moviepy.video.io.ImageSequenceClip
            from moviepy.editor import vfx
            clip = []
            import cv2
            for task_ind, (subtask, video) in enumerate(zip(eval_sequence, videos)):
                for img_ind, img in enumerate(video):
                    cv2.putText(img.copy(),
                                f'{task_ind}: {subtask}',
                                (10, 180),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5,
                                (0, 0, 0),
                                1,
                                2)
                    video[img_ind] = img
                clip.extend(video)
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(clip, fps=30)
            clip.write_videofile(f"calvin_seq{seq_ind}.mp4")


    return results


def evaluate_sequence(env, model, task_checker, initial_state, eval_sequence,
                      val_annotations, save_video):
    """
    Args:
        env: an instance of CALVIN_ENV
        model: an instance of CalvinBaseModel
        task_checker: an indicator of whether the current task is completed
        initial_state: a tuple of `robot_obs` and `scene_obs`
            see: https://github.com/mees/calvin/blob/main/dataset/README.md#state-observation
        eval_sequence: a list indicates the instruction chain
        val_annotations: a dictionary of task instructions
        save_video: a boolean indicates whether to save the video

    Returns:
        success_counter: an integer indicates the number of tasks completed
        video_aggregator: a list of lists of images that shows the trajectory
            of the robot

    """
    robot_obs, scene_obs = get_env_state_for_initial_condition(initial_state)
    env.reset(robot_obs=robot_obs, scene_obs=scene_obs)

    success_counter, video_aggregators = 0, []
    for subtask in eval_sequence:
        # get lang annotation for subtask
        success, video = rollout(env, model, task_checker, subtask)
        video_aggregators.append(video)

        if success:
            success_counter += 1
        else:
            return success_counter, video_aggregators
    return success_counter, video_aggregators


def rollout(env, model, task_oracle, subtask):
    """
    Run the actual rollout on one subtask (which is one natural language instruction).

    Args:
        env: an instance of CALVIN_ENV
        model: an instance of CalvinBaseModel
        task_oracle: an indicator of whether the current task is completed
        subtask: a string indicates the task name

    Returns:
        Success/Fail: a boolean indicates whether the task is completed
        video: a list of images that shows the trajectory of the robot
    """
    video = [] # show video for debugging
    obs = env.get_obs()

    model.reset()
    start_info = env.get_info()

    print('------------------------------')
    video.append(obs["rgb_obs"]["rgb_static"])

    n_action_steps = model.policy_cfg.n_action_steps
    pbar = tqdm(range(EP_LEN // n_action_steps))
    for step in pbar:
        obs = prepare_visual_states(obs, env, model.policy_cfg.n_obs_steps)
        obs = prepare_proprio_states(obs, env, model.policy_cfg.n_obs_steps)
        trajectory = model.step(obs)
        for act_ind in range(min(trajectory.shape[1], n_action_steps)):
            # calvin_env executes absolute action in the format of:
            # [[x, y, z], [euler_x, euler_y, euler_z], [open]]
            curr_action = [
                trajectory[0, act_ind, :3],
                trajectory[0, act_ind, 3:6],
                trajectory[0, act_ind, [6]]
            ]
            pbar.set_description(f"step: {step}")
            curr_proprio = obs['proprio']
            obs, _, _, current_info = env.step(curr_action)
            obs['proprio'] = curr_proprio

            # check if current step solves a task
            current_task_info = task_oracle.get_task_info_for_set(
                start_info, current_info, {subtask}
            )

            video.append(obs["rgb_obs"]["rgb_static"])

            if len(current_task_info) > 0:
                return True, video

    return False, video


def get_calvin_gripper_loc_bounds(args):
    with open(args.calvin_gripper_loc_bounds, "r") as stream:
       bounds = yaml.safe_load(stream)
       min_bound = bounds['act_min_bound'][:3]
       max_bound = bounds['act_max_bound'][:3]
       gripper_loc_bounds = np.stack([min_bound, max_bound])

    return gripper_loc_bounds


def main(args):

    # These location bounds are extracted from every episode in play trajectory
    if args.calvin_gripper_loc_bounds is not None:
        args.calvin_gripper_loc_bounds = get_calvin_gripper_loc_bounds(args)

    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # evaluate a custom model
    workspace = get_workspace(args)
    if workspace.cfg.training.use_ema:
        model = create_model(args, workspace.ema_model, workspace.cfg)
    else:
        model = create_model(args, workspace.model, workspace.cfg)

    sequence_indices = [
        i for i in range(args.local_rank, NUM_SEQUENCES, int(os.environ["WORLD_SIZE"]))
    ]

    env = make_env(args.calvin_dataset_path, show_gui=False)
    evaluate_policy(model, env,
                    conf_dir=Path(args.calvin_model_path) / "conf",
                    eval_log_dir=args.base_log_dir,
                    sequence_indices=sequence_indices,
                    save_video=args.save_video)

    results, sequence_inds = collect_results(args.base_log_dir)
    str_results = (
        " ".join([f"{i + 1}/5 : {v * 100:.1f}% |"
        for i, v in enumerate(count_success(results))]) + "|"
    )
    print(f'Load {len(results)}/1000 episodes...')
    print(str_results + "\n")

    del env
    gc.collect()


def get_workspace(args):
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy', 'config', f'{args.config_name}.yaml'))
    cfg = OmegaConf.load(config_path)
    task_config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy', 'config', 'task', 'calvin.yaml'))
    task_cfg = OmegaConf.load(task_config_path)
    cfg.task = task_cfg

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg, args.checkpoint_dir)
    lastest_ckpt_path = workspace.get_checkpoint_path()
    if lastest_ckpt_path.is_file():
        print(f"Resuming from checkpoint {lastest_ckpt_path}")
        workspace.load_checkpoint(path=lastest_ckpt_path)
    return workspace


if __name__ == "__main__":
    args = Arguments().parse_args()
    args.local_rank = int(os.environ["LOCAL_RANK"])

    # DDP initialization
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    main(args)
