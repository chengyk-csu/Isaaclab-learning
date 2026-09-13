import gymnasium as gym

gym.register(
    id="MyIsaac-Reach-Franka-IK-Rel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            "my_tasks.franka_reach.franka_reach_env_cfg:MyFrankaReachEnvCfg",
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