# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import gymnasium as gym
import pathlib
import sys

sys.path.insert(0, f"{pathlib.Path(__file__).parent.parent}")
from list_envs import import_packages  # noqa: F401

sys.path.pop(0)

tasks = []
for task_spec in gym.registry.values():
    if "Unitree" in task_spec.id and "Isaac" not in task_spec.id:
        tasks.append(task_spec.id)

import argparse

import argcomplete

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, choices=tasks, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
argcomplete.autocomplete(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import inspect
import os
import shutil
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner  # TODO: Consider printing the experiment name in the terminal.

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # This way, the Ray Tune workflow can extract experiment name.
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    agent_cfg_dict = agent_cfg.to_dict()
    if "obs_groups" not in agent_cfg_dict or agent_cfg_dict["obs_groups"] is None:
        agent_cfg_dict["obs_groups"] = {"policy": ["policy"], "critic": ["critic"]}
    runner = OnPolicyRunner(env, agent_cfg_dict, log_dir=log_dir, device=agent_cfg.device)
    # Attach AMP discriminator and expert buffer when amp_algorithm is present
    if hasattr(agent_cfg, "amp_algorithm"):
        try:
            from unitree_rl_lab.tasks.mimic.amp.core import AmpExpertBuffer, AMPDiscriminator, amp_discriminator_loss
            import torch

            # build expert buffer
            expert_file = agent_cfg.amp_algorithm.expert_motion_file
            expert_fps = getattr(agent_cfg.amp_algorithm, "expert_motion_fps", None)
            env.amp_expert_buffer = AmpExpertBuffer(expert_file, motion_fps=expert_fps, device=agent_cfg.device)

            # determine state dim from environment robot asset
            try:
                robot_asset = env.unwrapped.scene["robot"]
                num_joints = int(robot_asset.data.joint_pos.shape[1])
            except Exception:
                num_joints = int(getattr(agent_cfg.amp_algorithm, "amp_state_dim", 29))

            state_dim = 1 + num_joints * 2
            amp_disc = AMPDiscriminator(state_dim, hidden_dims=agent_cfg.amp_algorithm.discriminator_hidden_dims, activation=agent_cfg.amp_algorithm.discriminator_activation)
            amp_disc.to(agent_cfg.device)
            amp_opt = torch.optim.Adam(amp_disc.parameters(), lr=agent_cfg.amp_algorithm.discriminator_learning_rate)

            # attach to env so reward-term can access
            env.amp_discriminator = amp_disc
            env.amp_discriminator_opt = amp_opt
            env.amp_style_scale = getattr(agent_cfg.amp_algorithm, "style_reward_scale", 0.0) * getattr(agent_cfg.amp_algorithm, "reward_ratio", 1.0)
            env.amp_recent_transitions = []

            # monkeypatch runner.alg.update to alternate discriminator updates after PPO update
            if hasattr(runner, "alg") and hasattr(runner.alg, "update"):
                orig_update = runner.alg.update

                def wrapped_update(*args, **kwargs):
                    # call original PPO update (policy and value)
                    result = orig_update(*args, **kwargs)

                    # then perform discriminator update if enough policy transitions
                    try:
                        # sample expert transitions
                        batch_size = int(getattr(agent_cfg.amp_algorithm, "discriminator_batch_size", 256))
                        device = amp_disc.next_device if hasattr(amp_disc, "next_device") else agent_cfg.device
                        expert_trans = env.amp_expert_buffer.sample_transition(batch_size, step_dt=1.0 / env.amp_expert_buffer.motion_fps)
                        # sample policy transitions from recent buffer
                        policy_buf = getattr(env, "amp_recent_transitions", [])
                        if len(policy_buf) * getattr(env, "num_envs", 1) < batch_size:
                            return result

                        num_envs = policy_buf[0].shape[0]
                        idx = torch.randint(0, len(policy_buf), (batch_size,))
                        env_idx = torch.randint(0, num_envs, (batch_size,))
                        
                        policy_trans = torch.stack([policy_buf[i][e] for i, e in zip(idx, env_idx)], dim=0).to(agent_cfg.device)

                        # move expert transitions to device
                        expert_trans = expert_trans.to(agent_cfg.device)

                        # compute BCE loss
                        loss = amp_discriminator_loss(amp_disc, expert_trans, policy_trans)

                        # optional gradient penalty
                        gp_coef = float(getattr(agent_cfg.amp_algorithm, "gradient_penalty_coef", 0.0))
                        if gp_coef > 0.0:
                            # interpolate
                            eps = torch.rand(batch_size, 1, device=agent_cfg.device)
                            interp = eps * expert_trans + (1 - eps) * policy_trans
                            interp.requires_grad_(True)
                            logits = amp_disc(interp)
                            grads = torch.autograd.grad(outputs=logits.sum(), inputs=interp, create_graph=True)[0]
                            grads = grads.view(batch_size, -1)
                            grad_norm = grads.norm(2, dim=1)
                            gp = ((grad_norm - 1.0) ** 2).mean()
                            loss = loss + gp_coef * gp

                        amp_opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(amp_disc.parameters(), float(getattr(agent_cfg.amp_algorithm, "max_grad_norm", 1.0)))
                        amp_opt.step()
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"[WARN] AMP Discriminator update failed: {e}")

                    return result

                runner.alg.update = wrapped_update
        except Exception:
            print("[WARN] AMP initialization failed; continuing without AMP discriminator.")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    export_deploy_cfg(env.unwrapped, log_dir)
    # copy the environment configuration file to the log directory
    shutil.copy(
        inspect.getfile(env_cfg.__class__),
        os.path.join(log_dir, "params", os.path.basename(inspect.getfile(env_cfg.__class__))),
    )

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
