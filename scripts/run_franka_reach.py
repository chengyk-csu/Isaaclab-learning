import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="My first manager-based environment."
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=4,
)
parser.add_argument(
    "--task",
    type=str,
    default="MyIsaac-Reach-Franka-IK-Rel-v0",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import my_tasks
from isaaclab_tasks.utils import parse_env_cfg

def main():
    task_name = args_cli.task
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
    robot = env.unwrapped.scene["robot"]
    ee_body_ids, ee_body_names = robot.find_bodies(
        "panda_hand"
    )
    ee_body_id = ee_body_ids[0]
    actions = torch.zeros(
        env.unwrapped.num_envs,
        env.unwrapped.action_manager.total_action_dim,
        device=env.unwrapped.device,
    )
    actions[:, 0] = 0.2
    initial_ee_pos = robot.data.body_pos_w[
        :, ee_body_id, :
    ].clone()
    for step in range(30):
        with torch.inference_mode():
            obs, reward, terminated, truncated, info = env.step(actions)
    final_ee_pos = robot.data.body_pos_w[
        :, ee_body_id, :
    ].clone()
    displacement = final_ee_pos - initial_ee_pos
    distance = torch.linalg.norm(
        displacement,
        dim=1,
    )
    print("Task:", task_name)
    print("Action scale:", env_cfg.actions.arm_action.scale)

    print(
        "Mean EE displacement:",
        distance.mean().item(),
    )
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()