import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="My first Isaac Lab simulation."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab.managers import SceneEntityCfg

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
    sim.reset()
    robot = scene["robot"]


    print("[INFO] Franka initialized!")

    print("Joint names:")
    print(robot.joint_names)

    print("Number of joints:")
    print(robot.num_joints)

    print(
        "joint_pos shape:",
        robot.data.joint_pos.shape,
    )
    print(
        "joint_vel shape:",
        robot.data.joint_vel.shape,
    )

    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["panda_joint.*"],
        body_names=["panda_hand"],
    )
    robot_entity_cfg.resolve(scene)

    arm_joint_pos = robot.data.joint_pos[
        :, robot_entity_cfg.joint_ids
    ]
    arm_joint_vel = robot.data.joint_vel[
        :, robot_entity_cfg.joint_ids
    ]

    print(robot.body_names)

    ee_body_id = robot_entity_cfg.body_ids[0]
    ee_pos_w = robot.data.body_pos_w[
        :, ee_body_id
    ]
    ee_quat_w = robot.data.body_quat_w[
        :, ee_body_id
    ]

    print("All joint names:")
    print(robot.joint_names)

    print("Arm joint IDs:")
    print(robot_entity_cfg.joint_ids)

    print("Arm joint pos shape:")
    print(arm_joint_pos.shape)

    print("EE position shape:")
    print(ee_pos_w.shape)

    print("EE orientation shape:")
    print(ee_quat_w.shape)

    sim_dt = sim.get_physics_dt()
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

if __name__ == "__main__":
    main()
    simulation_app.close()