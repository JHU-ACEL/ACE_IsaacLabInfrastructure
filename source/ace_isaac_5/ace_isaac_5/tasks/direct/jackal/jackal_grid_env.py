# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import random
import torch
from collections.abc import Sequence
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg, AssetBaseCfg, AssetBase
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, spawn_from_usd
from isaaclab.sensors import TiledCamera

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils


from .jackal_grid_env_cfg import JackalGridEnvCfg

from .terrain_utilities.terrain_utils import TerrainManager


# SPHERE_MARKER_CFG = VisualizationMarkersCfg(
#     markers={
#         "sphere": sim_utils.SphereCfg(
#             radius=0.2,
#             visual_material=sim_utils.PreviewSurfaceCfg(
#                 diffuse_color=(0.0, 1.0, 1.0)),
#         ),
#     }
# )


GOAL_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.CuboidCfg(
            size=(0.25, 0.25, 1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)



def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
                "forward": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                "command": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                    scale=(0.25, 0.25, 0.5),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
        },
    )
    return VisualizationMarkers(cfg=marker_cfg)


#source /home/bchien1/IsaacSim/_build/linux-x86_64/release/setup_conda_env.sh

class JackalGridEnv(DirectRLEnv):
    cfg: JackalGridEnvCfg

    def __init__(self, cfg: JackalGridEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        print(self.dof_idx)

    def _setup_scene(self):

        # Device
        self.gpu = "cuda:0"

        # Scene Assets and Sensors
        self.robot = Articulation(self.cfg.robot_cfg)
        self.robot_camera = TiledCamera(self.cfg.tiled_camera)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg(size=(10000.0, 10000.0)))

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # Add sensors and articulations to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["robot_camera"] = self.robot_camera

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Arrow Markers Initialization (For Debugging/Visualization Purposes)
        self.arrows = define_markers()
        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()  
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()
        self.commands = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()

        self.marker_locations = torch.zeros((self.cfg.scene.num_envs, 3)).cuda()
        self.marker_offset = torch.zeros((self.cfg.scene.num_envs, 3), device=self.gpu)
        self.marker_offset[:, 2] = 0.5

        self.forward_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4), device=self.gpu)
        self.command_marker_orientations = torch.zeros((self.cfg.scene.num_envs, 4), device=self.gpu)
        self.up_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()  
        self.yaws = torch.zeros((self.cfg.scene.num_envs, 1)).cuda()

        # Goal Markers Initialization
        self.target_spawns = torch.zeros(self.cfg.scene.num_envs, 3, device=self.gpu)
        cube_cfg = GOAL_MARKER_CFG.copy()
        cube_cfg.prim_path = "/Visuals/Command/position_goal"
        self.goal_markers = VisualizationMarkers(cfg=cube_cfg)
        self.goal_markers.set_visibility(True)
        self.goal_radius = 5.0

        # Data structure to store observation history
        self.history_len = 5
        self._camera_hist: torch.Tensor | None = None

    def _get_goal_vec_normalized(self):
    
        goal_vec = self.target_spawns - self.robot.data.root_pos_w  
        return goal_vec/torch.linalg.norm(goal_vec, dim=-1, keepdim=True)


    def _update_marker_yaws(self):

        self.commands = self._get_goal_vec_normalized()

        ratio = self.commands[:,1]/(self.commands[:,0]+1E-8)
        gzero = torch.where(self.commands > 0, True, False)
        lzero = torch.where(self.commands < 0, True, False)
        plus = lzero[:,0]*gzero[:,1]
        minus = lzero[:,0]*lzero[:,1]
        offsets = torch.pi*plus - torch.pi*minus
        self.yaws = torch.atan(ratio).reshape(-1,1) + offsets.reshape(-1,1)


    def _visualize_markers(self):
        
        all_envs = torch.arange(self.cfg.scene.num_envs)
        indices = torch.hstack((torch.zeros_like(all_envs), torch.ones_like(all_envs)))

        self._update_marker_yaws()
        self.marker_locations = self.robot.data.root_pos_w
        self.forward_marker_orientations = self.robot.data.root_quat_w
        self.command_marker_orientations = math_utils.quat_from_angle_axis(self.yaws, self.up_dir).squeeze()
        loc = self.marker_locations + self.marker_offset
        loc = torch.vstack((loc, loc))
        rots = torch.vstack((self.forward_marker_orientations, self.command_marker_orientations))

        self.arrows.visualize(loc, rots, marker_indices=indices)
        self.goal_markers.visualize(self.target_spawns, marker_indices=indices)


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()
        self._visualize_markers()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:

        camera_data = self.robot_camera.data.output["rgb"] / 255.0
        # normalize the camera data for better training results
        mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        camera_data -= mean_tensor

        # On first call, fill history with the current frame
        if self._camera_hist is None:
            # repeat the first frame history_len times

            camera_data = camera_data.unsqueeze(1)
            camera_data = camera_data.repeat(1, self.history_len, 1, 1, 1)
            self._camera_hist = camera_data

        else:
            # drop oldest frame and append newest
            # _camera_hist[:, 1:] are t-3…t, so cat with new frame at dim=1
            new = camera_data.unsqueeze(1)   # (N,1,H,W,C)
            self._camera_hist = torch.cat([self._camera_hist[:, 1:], new], dim=1)

        return {"policy": self._camera_hist.clone()}


    def _get_rewards(self) -> torch.Tensor:

        goal_vec = self._get_goal_vec_normalized() # torch.Size([N, 3]) 
         
        forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w,
                                        self.robot.data.FORWARD_VEC_B) # torch.Size([N, 3]) 
        
        alignment = torch.sum(forwards * goal_vec, dim=-1, keepdim=True) # torch.Size([N, 1]), batchwise dot product 
                                                                        # between the vectors in the tensors goal_vec and forwards
        angle = torch.acos(alignment.clamp(-1.0, 1.0)) # torch.Size([N, 1]) 
        # Since the vectors are unit vectors, the dot product is equal to cos(theta), so take the inverse cosine

        vel = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1) # torch.Size([N, 1])
        threshold = math.pi / 9.0
        mask = (angle <= threshold).to(torch.float32) # torch.Size([N, 1])
        forward_reward = vel * mask # torch.Size([N, 1])

        base_reward = forward_reward*torch.exp(5*alignment) # torch.Size([N, 1])

        # arrival bonus
        dist = torch.linalg.norm(
            self.target_spawns - self.robot.data.root_pos_w, dim=-1, keepdim=True
        ) # torch.Size([N, 1])


        arrived_mask = dist < 0.5                                                   # BoolTensor (N,1)
        arrival_bonus = arrived_mask.to(torch.float32) * (1.0 * base_reward)        # torch.Size([N, 1])
        total_reward = base_reward + arrival_bonus                                  # torch.Size([N, 1])

        import pdb; pdb.set_trace()

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        time_out = self.episode_length_buf >= (self.max_episode_length - 1)

        dist_to_goal = torch.linalg.norm(
            self.target_spawns - self.robot.data.root_pos_w, dim=-1
        )

        reached = dist_to_goal < 0.5

        # if reached:
        #     print("GOAL REACHED: TIMEOUT HERE")

        return reached, time_out


    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        self.robot.write_root_state_to_sim(default_root_state, env_ids)
    
        # Have half_span range from math.pi/7.0 to math.pi/4.0
        #half_span = math.pi/4.0     
        # half_span = random.uniform(math.pi/7.0, math.pi/4.0)
        # #

        # angles = torch.empty(len(env_ids), device=self.gpu).uniform_(-half_span, half_span)

        N = len(env_ids)
        device = self.gpu
        # 1) sample absolute angles in [π/6, π/4]
        min_ang, max_ang = math.pi/4.0, math.pi/2.0
        mags = torch.empty(N, device=device).uniform_(min_ang, max_ang)

        # 2) pick random sign ±1 for each env
        signs = torch.where(torch.rand(N, device=device) < 0.5,
                            1.0,  # positive branch
                        -1.0)  # negative branch

        # 3) combine
        angles = mags * signs  # shape (N,)

        # half_span = math.pi/8.0     
        # angles = torch.empty(len(env_ids), device=self.gpu).uniform_(half_span, half_span)

        targets = default_root_state[:, :3].clone()
        targets[:, 0] = targets[:, 0] + self.goal_radius * torch.cos(angles)
        targets[:, 1] = targets[:, 1] + self.goal_radius * torch.sin(angles)   
        self.target_spawns[env_ids] = targets

        self._visualize_markers()

