from typing import Optional
import logging

import torch
import numpy as np
import einops

# This is for using the locally installed repo clone when using slurm
from calvin_utils.online_evaluation_calvin.evaluate_utils import convert_action
from calvin_utils.utils_with_calvin import relative_to_absolute


logger = logging.getLogger(__name__)


def create_model(args, policy, policy_cfg):
    model = ModelWrapperForEvaluation(args, policy, policy_cfg)

    return model


class ModelWrapperForEvaluation:
    """A wrapper for the model, which handles
            1. Model initialization
            2. Model inference
            3. Action post-processing
                - quaternion to Euler angles
                - relative to absolute action
    """
    def __init__(self, args, policy, policy_cfg):
        self.args = args
        self.policy = policy
        self.policy_cfg = policy_cfg 
        self.reset()

    def reset(self):
        """Set model to evaluation mode.
        """
        device = self.args.device
        self.policy.eval()
        self.policy = self.policy.to(device)

    def step(self, obs):
        """
        Args:
            dp_obs: a dictionary of observations
                - rgb_obs: a dictionary of RGB images
            proprio: a np.array of gripper poses

        Returns:
            action: predicted action
        """
        device = self.args.device

        # Organize inputs
        rgbs = obs["dp_obs"]["rgb_obs"]["rgb_static"]  # [T, H, W, 3]

        rgbs = torch.as_tensor(rgbs).to(device)

        # Crop the images.  See Line 165-166 in datasets/dataset_calvin.py
        nobs_steps, ncam = rgbs.shape[:2]
        rgbs = einops.rearrange(rgbs, "T H W C -> 1 T C H W")

        # history of actions
        gripper = torch.as_tensor(obs["proprio"]).to(device).unsqueeze(0)

        # create obs dict
        obs_dict = {
            'image': rgbs.float(), # nobs, npts, 6
            'agent_pos': gripper.float(), # nobs, D_pos
        }

        # run policy
        with torch.no_grad():
            trajectory = self.policy.predict_action(obs_dict)['action_pred']

        # Convert quaternion to Euler angles
        trajectory = convert_action(trajectory)

        if bool(self.args.relative_action):
            # Convert quaternion to Euler angles
            gripper = convert_action(gripper[:, [-1], :])
            # Convert relative action to absolute action
            trajectory = relative_to_absolute(trajectory, gripper)

        # Bound final action by CALVIN statistics
        if self.args.calvin_gripper_loc_bounds is not None:
            trajectory[:, :, :3] = np.clip(
                trajectory[:, :, :3],
                a_min=self.args.calvin_gripper_loc_bounds[0].reshape(1, 1, 3),
                a_max=self.args.calvin_gripper_loc_bounds[1].reshape(1, 1, 3)
            )

        return trajectory
