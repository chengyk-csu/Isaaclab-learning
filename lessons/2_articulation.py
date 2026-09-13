import argparse
from isaaclab.app import AppLauncher
import torch

parser = argparse.ArgumentParser(
    description="My first Isaac Lab simulation."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


from isaaclab.sim import SimulationCfg, SimulationContext
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab_assets import CARTPOLE_CFG

def main():
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(
        eye=[2.5, 2.5, 2.5],
        target=[0.0, 0.0, 0.0],
    )
    cartpole_cfg = CARTPOLE_CFG.copy()
    cartpole_cfg.prim_path = "/World/Robot"
    robot = Articulation(cfg=cartpole_cfg)

    sim.reset()
    cart_joint_ids, cart_joint_names = robot.find_joints("slider_to_cart")
    print("cart_joint_ids:", cart_joint_ids)
    print("cart_joint_names:", cart_joint_names)
    sim_dt = sim.get_physics_dt()
    print("Joint names:", robot.joint_names)
    print("Number of joints:", robot.num_joints)
    print("Joint position:")
    print(robot.data.joint_pos)
    print("Joint velocity:")
    print(robot.data.joint_vel)
    print("joint_pos shape:", robot.data.joint_pos.shape)
    print("joint_vel shape:", robot.data.joint_vel.shape)
    print("[INFO] Simulation is ready!")

    effort = torch.tensor(
        [[5.0]],
        device=sim.device,
    )
    #count = 0 #临时
    while simulation_app.is_running():
        robot.set_joint_effort_target(
            effort,
            joint_ids=cart_joint_ids,
        )
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        #临时
        # count += 1
        # if count % 100 == 0:
        #     print("joint_pos:", robot.data.joint_pos)
        #     print("joint_vel:", robot.data.joint_vel)

if __name__ == "__main__":
    main()
    simulation_app.close()