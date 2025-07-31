    # def compute(self, inputs, role):

    #     obs = unflatten_tensorized_space(self.observation_space, inputs["states"])
    #     obs = obs.permute(0, 1, 4, 2, 3)
    #     N, T, C, H, W = obs.shape
    #     obs = obs.contiguous().view(N * T, C, H, W)
    #     obs = obs.to(self.device)

    #     feats = self.features_extractor(obs)
    #     feats = feats.view(N, T*self.feat_dim)
    #     #import pdb; pdb.set_trace()
    #     shared = self.fc1(feats)

    #     if role == "policy":
    #         mean_action = self.mean_layer(shared)
    #         return mean_action, self.log_std_parameter, {}

    #     elif role == "value":
    #         value = self.value_layer(shared)
    #         return value, {}


# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import ace_isaac_5.tasks  # noqa: F401

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"



''' Custom NN Network Defined Here (Pytorch Style) '''
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from skrl.utils.spaces.torch import unflatten_tensorized_space


# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.observation_space = observation_space
        self.device = device
        self.steps, self.camera_h, self.camera_w, self.channels = observation_space.shape
        self.action_space = action_space.shape[0]

        self.features_extractor = nn.Sequential(
            nn.Conv2d(in_channels=self.channels,  
                      out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        self.features_extractor = self.features_extractor.to(device)

        with torch.no_grad():
            # dummy state: batch=1, channels, height, width
            dummy = torch.zeros(1,
                                self.channels,
                                self.camera_h,
                                self.camera_w,
                                device=device)
            feat_dim = self.features_extractor(dummy).shape[1]

        print(f"FEATURE DIMENSION: {feat_dim}")
            
        self.fc1 = nn.Sequential(nn.Linear(self.steps*feat_dim, 512),
                                 nn.ELU())

        self.mean_layer = nn.Linear(512, self.action_space)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.action_space))

        self.value_layer = nn.Linear(512, 1)
        self.to(device)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
        

    # def compute(self, inputs, role):

    #     obs = unflatten_tensorized_space(self.observation_space, inputs["states"])
    #     obs = obs.permute(0, 1, 4, 2, 3)
    #     N, T, C, H, W = obs.shape
    #     obs = obs.contiguous().view(N * T, C, H, W)
    #     obs = obs.to(self.device)

    #     feats = self.features_extractor(obs)
    #     feats = feats.view(N, T*self.feat_dim)
    #     #import pdb; pdb.set_trace()
    #     shared = self.fc1(feats)

    #     if role == "policy":
    #         mean_action = self.mean_layer(shared)
    #         return mean_action, self.log_std_parameter, {}

    #     elif role == "value":
    #         value = self.value_layer(shared)
    #         return value, {}

    def compute(self, inputs, role):

        # inputs["states"] expected shape: (N, H, W, C)
        obs = unflatten_tensorized_space(self.observation_space, inputs["states"])
        per_frame_obs = list(torch.unbind(obs, dim=1))

        # per_frame_obs: list of 5 tensors, each shape (N, T, H, W, C)
        frame_feats = []
        for frame in per_frame_obs:
            # move to device & convert NHWC → NCHW
            x = frame.to(self.device).permute(0, 3, 1, 2)      # → (N, C, H, W)
            # run through your 2D‐CNN extractor
            feat = self.features_extractor(x)                 # → (N, feat_dim)
            frame_feats.append(feat)

        # frame_feats: list of 5 tensors of shape (N, feat_dim)
        flat_feats = torch.cat(frame_feats, dim=1)   # → (N, 5 * feat_dim)
        # print(f"FLAT FEATS SHAPE: {flat_feats.shape}")

        shared = self.fc1(flat_feats)           # → (N, 512)

        if role == "policy":
            mean_action = self.mean_layer(shared)
            return mean_action, self.log_std_parameter, {}

        elif role == "value":
            value = self.value_layer(shared)
            return value, {}
        
''' Custom definition ends Here '''


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # # specify directory for logging experiments
    # log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    # log_root_path = os.path.abspath(log_root_path)
    # print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # # specify directory for logging runs: {time-stamp}_{run_name}
    # log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    # print(f"Exact experiment name requested from command line: {log_dir}")
    # if agent_cfg["agent"]["experiment"]["experiment_name"]:
    #     log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # # set directory into agent config
    # agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    # agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # # update log_dir
    # log_dir = os.path.join(log_root_path, log_dir)

    # # dump the configuration into log-directory
    # dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    # dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    # dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    # if args_cli.video:
    #     video_kwargs = {
    #         "video_folder": os.path.join(log_dir, "videos", "train"),
    #         "step_trigger": lambda step: step % args_cli.video_interval == 0,
    #         "video_length": args_cli.video_length,
    #         "disable_logger": True,
    #     }
    #     print("[INFO] Recording videos during training.")
    #     print_dict(video_kwargs, nesting=4)
    #     env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # # configure and instantiate the skrl runner
    # # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    # runner = Runner(env, agent_cfg)

    # # load checkpoint (if specified)
    # if resume_path:
    #     print(f"[INFO] Loading model checkpoint from: {resume_path}")
    #     runner.agent.load(resume_path)

    # # run training
    # runner.run()


    device = env.device

    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)


    # instantiate the agent's models (function approximators).
    # PPO requires 2 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
    models = {}
    models["policy"] = Shared(env.observation_space, env.action_space, device)
    models["value"] = models["policy"]  # same instance: shared model


    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] = 64  # memory_size
    cfg["learning_epochs"] = 4
    cfg["mini_batches"] = 4  # 24 * 4096 / 24576
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 1.0e-04
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 1.0
    cfg["kl_threshold"] = 0
    cfg["rewards_shaper"] = None
    cfg["time_limit_bootstrap"] = False
    # cfg["state_preprocessor"] = RunningStandardScaler
    # cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging to TensorBoard and write checkpoints (in timesteps)
    cfg["experiment"]["write_interval"] = 60
    cfg["experiment"]["checkpoint_interval"] = 600
    cfg["experiment"]["directory"] = "runs/torch/Isaac-Jackal-v0"

    agent = PPO(models=models,
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 32000, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.train()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()