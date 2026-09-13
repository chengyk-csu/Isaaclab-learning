import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="My first Isaac Lab simulation."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=4,
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg

def main():
    task_name = "Isaac-Reach-Franka-IK-Rel-v0"
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
    print("Observation keys:")
    print(obs.keys())
    print(
        "Policy observation shape:",
        obs["policy"].shape,
    )
    print(
        "Action dimension:",
        env.unwrapped.action_manager.total_action_dim,
    )
    print(
        "Number of environments:",
        env.unwrapped.num_envs,
    )
    actions = torch.zeros(
        env.unwrapped.num_envs,
        env.unwrapped.action_manager.total_action_dim,
        device=env.unwrapped.device,
    )
    print(
        env.unwrapped.action_manager.total_action_dim
    )
    while simulation_app.is_running():
        with torch.inference_mode():
            obs, reward, terminated, truncated, info = env.step(actions)

if __name__ == "__main__":
    main()
    simulation_app.close()