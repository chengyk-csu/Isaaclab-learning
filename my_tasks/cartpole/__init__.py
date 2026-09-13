import gymnasium as gym

gym.register(
    id="MyIsaac-Cartpole-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point":
            "my_tasks.cartpole.cartpole_env_cfg:CartpoleEnvCfg",
    },
)