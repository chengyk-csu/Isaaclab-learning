import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_envs",
    type=int,
    default=4,
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import gymnasium as gym
import my_tasks
from isaaclab_tasks.utils import parse_env_cfg

def main():
    task_name = "MyIsaac-Cartpole-v0"
    env_cfg = parse_env_cfg(
        task_name,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env = gym.make(
        task_name,
        cfg=env_cfg,
    )
    obs, info = env.reset()
    print(
        "Observation shape:",
        obs["policy"].shape,
    )
    print(
        "Number of environments:",
        env.unwrapped.num_envs,
    )
    print(
        "Action dimension:",
        env.unwrapped.action_manager.total_action_dim,
    )
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.randn(
                env.unwrapped.num_envs,
                env.unwrapped.action_manager.total_action_dim,
                device=env.unwrapped.device,
            )
            obs, reward, terminated, truncated, info = env.step(actions)
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()