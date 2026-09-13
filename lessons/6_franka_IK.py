import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="My first Isaac Lab simulation."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.controllers import (
    DifferentialIKController,
    DifferentialIKControllerCfg,
)
from isaaclab.utils.math import subtract_frame_transforms

@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0
        ),
    )
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

def main():
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(
        eye=[3.0, 3.0, 2.5],
        target=[0.0, 0.0, 0.7],
    )
    scene_cfg = FrankaSceneCfg(
        num_envs=4,
        env_spacing=2.5,
    )
    scene = InteractiveScene(scene_cfg)
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )

    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg,
        num_envs=scene.num_envs,
        device=sim.device,
    )

    ee_goal = torch.tensor(
        [
            [0.5, 0.3, 0.6, 0.707, 0.0, 0.707, 0.0]
        ],
        device=sim.device,
    )

    ee_goal = ee_goal.repeat(
        scene.num_envs,
        1,
    )

    sim.reset()
    robot = scene["robot"]
    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["panda_joint.*"],
        body_names=["panda_hand"],
    )
    robot_entity_cfg.resolve(scene)
    ee_body_id = robot_entity_cfg.body_ids[0]
    ee_jacobi_idx = ee_body_id - 1

    sim_dt = sim.get_physics_dt()

    diff_ik_controller.reset()
    diff_ik_controller.set_command(ee_goal)
    while simulation_app.is_running():
        jacobian = robot.root_physx_view.get_jacobians()[
            :,
            ee_jacobi_idx,
            :,
            robot_entity_cfg.joint_ids,
        ]
        ee_pose_w = robot.data.body_pose_w[
            :,
            ee_body_id,
        ]
        root_pose_w = robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )
        joint_pos = robot.data.joint_pos[
            :,
            robot_entity_cfg.joint_ids,
        ]
        joint_pos_des = diff_ik_controller.compute(
            ee_pos_b,
            ee_quat_b,
            jacobian,
            joint_pos,
        )
        robot.set_joint_position_target(
            joint_pos_des,
            joint_ids=robot_entity_cfg.joint_ids,
        )
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

if __name__ == "__main__":
    main()
    simulation_app.close()