from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.reach.config.franka.agents.rsl_rl_ppo_cfg import (
    FrankaReachPPORunnerCfg,
)


@configclass
class MyFrankaReachPPORunnerCfg(FrankaReachPPORunnerCfg):

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "my_franka_reach"