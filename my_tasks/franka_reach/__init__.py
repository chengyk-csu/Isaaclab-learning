import gymnasium as gym
from . import agents

gym.register(
    id="MyIsaac-Reach-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            "my_tasks.franka_reach.franka_reach_env_cfg:MyFrankaReachEnvCfg",

        "rsl_rl_cfg_entry_point":
            "my_tasks.franka_reach.agents.rsl_rl_ppo_cfg:MyFrankaReachPPORunnerCfg",
    },
)
gym.register(
    id="MyIsaac-Reach-Franka-IK-Rel-Scale02-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            "my_tasks.franka_reach.franka_reach_env_cfg:MyFrankaReachEnvCfgScale02",
    },
)

gym.register(
    id="MyIsaac-Reach-Franka-IK-Rel-ExpReward-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            "my_tasks.franka_reach.franka_reach_env_cfg:MyFrankaReachExpRewardEnvCfg",

        "rsl_rl_cfg_entry_point":
            "my_tasks.franka_reach.agents.rsl_rl_ppo_cfg:MyFrankaReachPPORunnerCfg",
    },
)