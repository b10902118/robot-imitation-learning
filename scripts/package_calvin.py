from typing import List, Optional
from pathlib import Path
import os
import pickle

import tap
import cv2
import numpy as np
import torch
from PIL import Image


class Arguments(tap.Tap):
    save_path: str = './data/calvin/packaged_D_D'
    root_dir: str = './calvin/dataset/task_D_D'
    tasks: List[str] = ["rotate_red_block_right",
                        "rotate_blue_block_right",
                        "rotate_pink_block_right"]
    split: str = 'training'  # [training, validation]


def process_datas(datas):
    """Fetch and drop datas to make a trajectory

    Args:
        datas: a dict of the datas to be saved/loaded
            - static_rgb: a list of nd.arrays with shape (height, width, 3)
            - gripper_rgb: a list of nd.arrays with shape (height, width, 3)
            - proprios: a list of nd.arrays with shape (7,)

    Returns:
        the episode item: [
            [frame_ids],
            [obs_tensors],  # wrt frame_ids, (n_cam, 2, 3, 256, 256)
                obs_tensors[i][:, 0] is RGB, obs_tensors[i][:, 1] is XYZ
            [action_tensors],  # wrt frame_ids, (1, 8)
            [camera_dicts],
            [gripper_tensors],  # wrt frame_ids, (1, 8)
            [trajectories]  # wrt frame_ids, (N_i, 8)
            [annotation_ind] # wrt frame_ids, (1,)
        ]
    """
    # upscale gripper camera
    h, w = datas['static_rgb'][0].shape[:2]
    datas['gripper_rgb'] = [
        cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        for m in datas['gripper_rgb']
    ]
    static_rgb = np.stack(datas['static_rgb'], axis=0) # (traj_len, H, W, 3)
    gripper_rgb = np.stack(datas['gripper_rgb'], axis=0) # (traj_len, H, W, 3)
    rgb = np.stack([static_rgb, gripper_rgb], axis=1) # (traj_len, ncam, H, W, 3)
    rgb = rgb.transpose(0, 1, 4, 2, 3) # (traj_len, ncam, 3, H, W)

    # prepare camera_dicts
    camera_dicts = [{'front': (0, 0), 'wrist': (0, 0)}]

    # prepare gripper tensors
    gripper_tensors = torch.cat([
        torch.as_tensor(a, dtype=torch.float32).view(1, -1)
        for a in datas['proprios']
    ], dim=0).data.cpu().numpy()

    # prepare frame_ids
    frame_ids = [i for i in range(len(rgb))]

    # Save everything to disk
    state_dict = [
        frame_ids,
        rgb,
        gripper_tensors,
        camera_dicts,
        datas['annotation_id']
    ]

    return state_dict


def load_episode(root_dir, split, episode, datas, ann_id):
    """Load episode and process datas

    Args:
        root_dir: a string of the root directory of the dataset
        split: a string of the split of the dataset
        episode: a string of the episode name
        datas: a dict of the datas to be saved/loaded
            - static_rgb: a list of nd.arrays with shape (height, width, 3)
            - gripper_rgb: a list of nd.arrays with shape (height, width, 3)
            - proprios: a list of nd.arrays with shape (8,)
            - annotation_id: a list of ints
    """
    data = np.load(f'{root_dir}/{split}/{episode}')

    rgb_static = data['rgb_static']  # (200, 200, 3)
    rgb_gripper = data['rgb_gripper']  # (84, 84, 3)

    # Map gripper openess to [0, 1]
    proprio = np.concatenate([
        data['robot_obs'][:3],
        data['robot_obs'][3:6],
        (data['robot_obs'][[-1]] > 0).astype(np.float32)
    ], axis=-1)

    # Put them into a dict
    datas['static_rgb'].append(rgb_static)  # (200, 200, 3)
    datas['gripper_rgb'].append(rgb_gripper)  # (84, 84, 3)
    datas['proprios'].append(proprio)  # (8,)
    datas['annotation_id'].append(ann_id)  # int


def init_datas():
    datas = {
        'static_rgb': [],
        'gripper_rgb': [],
        'proprios': [],
        'annotation_id': []
    }
    return datas


def main(split, args):
    """
    CALVIN contains long videos of "tasks" executed in order
    with noisy transitions between them. The 'annotations' json contains
    info on how to segment those videos.

    Original CALVIN annotations:
    {
        'info': {
            'episodes': [],
            'indx': [(788072, 788136), (899273, 899337), (1427083, 1427147)]
                list of tuples indicating start-end of a task
        },
        'language': {
            'ann': list of str with len=17870, instructions,
            'task': list of str with len=17870, task names,
            'emb': array (17870, 1, 384)
        }
    }

    Save:
    state_dict = [
        frame_ids,  # [0, 1, 2...]
        rgb,  # tensor [len(frame_ids), ncam, 2, 3, 200, 200]
        gripper_tensors,  # [tensor(1, 8)]
        camera_dicts,  # [{'front': (0, 0), 'wrist': (0, 0)}]
        datas['annotation_id']  # [int]
    ]
    """
    annotations = np.load(
        f'{args.root_dir}/{split}/lang_annotations/auto_lang_ann.npy',
        allow_pickle=True
    ).item()

    for anno_ind, (start_id, end_id) in enumerate(annotations['info']['indx']):
        # Step 1. load episodes of the same task
        len_anno = len(annotations['info']['indx'])
        if args.tasks is not None and annotations['language']['task'][anno_ind] not in args.tasks:
            continue
        print(f'Processing {anno_ind}/{len_anno}, start_id:{start_id}, end_id:{end_id}')
        datas = init_datas()
        for ep_id in range(start_id, end_id + 1):
            episode = 'episode_{:07d}.npz'.format(ep_id)
            load_episode(
                args.root_dir,
                split,
                episode,
                datas,
                anno_ind
            )

        # Step 2. detect keyframes within the episode
        state_dict = process_datas(datas)

        # Step 3. determine scene
        scene = 'D'

        # Step 4. save to .dat file
        ep_save_path = f'{args.save_path}/{split}/{scene}+0/ann_{anno_ind}.npy'
        os.makedirs(os.path.dirname(ep_save_path), exist_ok=True)
        np.save(ep_save_path, state_dict)


if __name__ == "__main__":
    args = Arguments().parse_args()
    main(args.split, args)
