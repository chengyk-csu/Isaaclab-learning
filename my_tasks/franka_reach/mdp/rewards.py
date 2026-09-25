import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms

def position_command_error_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:

    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_pos_b = command[:, :3]
    target_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w,
        asset.data.root_quat_w,
        target_pos_b,
    )
    ee_pos_w = asset.data.body_pos_w[
        :,
        asset_cfg.body_ids[0],
    ]
    distance = torch.linalg.norm(
        ee_pos_w - target_pos_w,
        dim=1,
    )
    reward = torch.exp(
        -distance / std
    )

    return reward