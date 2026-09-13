import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="My first Isaac Lab simulation."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaaclab.sim import SimulationCfg, SimulationContext
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets import CARTPOLE_CFG


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0),
    )
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


def main():
    sim_cfg = sim_utils.SimulationCfg(
        device=args_cli.device
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(
        eye=[6.0, 6.0, 4.0],
        target=[0.0, 0.0, 1.0],
    )
    scene_cfg = CartpoleSceneCfg(
        num_envs=4,
        env_spacing=2.0,
    )
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["cartpole"]
    sim_dt = sim.get_physics_dt()
    cart_joint_ids, _ = robot.find_joints("slider_to_cart")
    efforts = torch.linspace(
        -5.0,
        5.0,
        steps=robot.num_instances,
        device=sim.device,
    )
    efforts = efforts.unsqueeze(1)
    print("num robots:", robot.num_instances)
    print("joint_pos shape:", robot.data.joint_pos.shape)
    print("action shape:", efforts.shape)

    while simulation_app.is_running():
        robot.set_joint_effort_target(
            efforts,
            joint_ids=cart_joint_ids,
        )
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

if __name__ == "__main__":
    main()
    simulation_app.close()