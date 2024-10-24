from typing import Dict
import pickle
import os

import torch

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from calvin_utils.dataset_base.dataset_calvin import CalvinDataset as CalvinDatasetBackend


# very hacky, use absolute path
TRAIN_SET_DIR = os.path.dirname(__file__).replace("diffusion_policy/dataset", "data/calvin/packaged_D_D/training")
VAL_SET_DIR = os.path.dirname(__file__).replace("diffusion_policy/dataset", "data/calvin/packaged_D_D/validation")
MAX_EPISODE_PER_TASK = -1
CAMERAS = ("front", )
IMAGE_RESCALE = "0.75,1.25"
RELATIVE_ACTION = True


class CalvinDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            n_obs_steps=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            ):
        super().__init__()
        self.task_name = task_name
        self.pad_before = pad_before
        self.pad_after = pad_after
        self._horizon = horizon
        self._n_obs_steps = n_obs_steps

        taskvar = [
            ("D", 0),
        ]
        self.backend_dataset = CalvinDatasetBackend(
            root=TRAIN_SET_DIR,
            instructions=None,
            taskvar=taskvar,
            horizon=horizon,
            nobs_step=n_obs_steps,
            cache_size=0,
            max_episodes_per_task=MAX_EPISODE_PER_TASK,
            num_iters=None,
            cameras=CAMERAS,
            training=True,
            image_rescale=tuple(
                float(x) for x in IMAGE_RESCALE.split(",")
            ),
            relative_action=RELATIVE_ACTION,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
        )

    def get_validation_dataset(self):
        taskvar = [
            ("D", 0),
        ]
        self.backend_dataset = CalvinDatasetBackend(
            root=VAL_SET_DIR,
            instructions=None,
            taskvar=taskvar,
            horizon=self._horizon,
            nobs_step=self._n_obs_steps,
            cache_size=0,
            max_episodes_per_task=MAX_EPISODE_PER_TASK,
            num_iters=None,
            cameras=CAMERAS,
            training=False,
            image_rescale=tuple(
                float(x) for x in IMAGE_RESCALE.split(",")
            ),
            relative_action=RELATIVE_ACTION,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
        )
        return self

    def get_normalizer(self, mode='limits', **kwargs):
        _horizon = self.backend_dataset._horizon
        _nobs_step = self.backend_dataset._nobs_step

        action, agent_pos = [], []
        for i in range(len(self.backend_dataset)):
            for _ in range(15):
                ep = self.backend_dataset[i]
                action.append(ep['action'])  # (horizon, D_action)
                agent_pos.append(ep['agent_pos'])  # (nobs_step, D_pos)
        action = torch.cat(action, dim=0)
        agent_pos = torch.cat(agent_pos, dim=0)

        data = {
            'action': action,
            'agent_pos': agent_pos,
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def __len__(self) -> int:
        return len(self.backend_dataset)

    def _sample_to_data(self, sample):
        agent_pos = sample['agent_pos'].float()
        image = sample['rgbs'].float() / 255.

        data = {
            'obs': {
                'image': image,
                'agent_pos': agent_pos, # nobs, D_pos
            },
            'action': sample['action'].float() # T, D_action
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.backend_dataset[idx]
        torch_data = self._sample_to_data(sample)
        return torch_data