from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.manipulation.reach.config.franka.ik_rel_env_cfg import (
    FrankaReachEnvCfg,
)


@configclass
class MyFrankaReachEnvCfg(FrankaReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # self.commands.ee_pose.ranges.pos_x = (0.45, 0.55)
        # self.commands.ee_pose.ranges.pos_y = (-0.10, 0.10)
        # self.commands.ee_pose.ranges.pos_z = (0.30, 0.40)

@configclass
class MyFrankaReachEnvCfgScale02(MyFrankaReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.actions.arm_action.scale = 0.2

from . import mdp

@configclass
class MyFrankaReachExpRewardEnvCfg(
    MyFrankaReachEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.end_effector_position_tracking_fine_grained.func = (
            mdp.position_command_error_exp
        )