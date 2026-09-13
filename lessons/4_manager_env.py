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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch
import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
)
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import (
    CartpoleSceneCfg,
)
import math
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    TerminationTermCfg as DoneTerm,
    SceneEntityCfg,
)
from isaaclab.envs import (
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
)


@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=100.0,
    )

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel
        )
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    alive = RewTerm(
        func=mdp.is_alive,
        weight=1.0,
    )
    terminating = RewTerm(
        func=mdp.is_terminated,
        weight=-2.0,
    )
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["cart_to_pole"],
            ),
            "target": 0.0,
        },
    )
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["slider_to_cart"],
            ),
        },
    )
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["cart_to_pole"],
            ),
        },
    )

@configclass
class TerminationsCfg:
    time_out = DoneTerm(
        func=mdp.time_out,
        time_out=True,
    )
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["slider_to_cart"],
            ),
            "bounds": (-3.0, 3.0),
        },
    )

@configclass
class EventCfg:
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["slider_to_cart"],
            ),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["cart_to_pole"],
            ),
            "position_range": (
                -0.25 * math.pi,
                0.25 * math.pi,
            ),
            "velocity_range": (
                -0.25 * math.pi,
                0.25 * math.pi,
            ),
        },
    )


@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    scene = CartpoleSceneCfg(
        num_envs=4,
        env_spacing=4.0,
    )
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    def __post_init__(self):
        self.episode_length_s = 5.0
        self.decimation = 2
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation


def main():
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)
    obs, info = env.reset()
    print("Observation keys:")
    print(obs.keys())
    print("Policy observation:")
    print(obs["policy"])
    print("Policy observation shape:")
    print(obs["policy"].shape)
    print("Action shape:")
    print(env.action_manager.action.shape)

    count = 0#暂时

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = torch.randn_like(
                env.action_manager.action
            )
            obs, reward, terminated, truncated, info = env.step(actions)
            count += 1
            if count % 60 == 0:
                print("obs shape:", obs["policy"].shape)
                print("action shape:", actions.shape)
                print("reward shape:", reward.shape)
                print("terminated shape:", terminated.shape)
                print("truncated shape:", truncated.shape)
                print("env 0 reward:", reward[0].item())


if __name__ == "__main__":
    main()
    simulation_app.close()