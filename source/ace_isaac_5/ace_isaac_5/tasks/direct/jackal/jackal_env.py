# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg, AssetBaseCfg, AssetBase
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane, spawn_from_usd
from isaaclab.sensors import TiledCamera

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils


from .jackal_env_cfg import JackalEnvCfg

from .terrain_utilities.terrain_utils import TerrainManager


SPHERE_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "sphere": sim_utils.SphereCfg(
            radius=0.2,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.0, 0.0)),
        ),
    }
)



class JackalEnv(DirectRLEnv):
    cfg: JackalEnvCfg

    def __init__(self, cfg: JackalEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)
        print(self.dof_idx)

    def _setup_scene(self):

        # Scene Assets and Sensors
        self.robot = Articulation(self.cfg.robot_cfg)
        self.robot_camera = TiledCamera(self.cfg.tiled_camera)

        # add ground plane
        #spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # Update the scene's attributes
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["tiled_camera"] = self.robot_camera

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=1000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


        # Terrain Manager
        self.terrainManager = TerrainManager(
            num_envs=self.cfg.scene.num_envs, 
            device="cuda:0",
            # terrain_usd_path = "/home/bchien1/ACE_IsaacLabInfrastructure/source/ace_isaac_5/ace_isaac_5/mars_terrain/terrain_only.usd",
            # rock_usd_path = "/home/bchien1/ACE_IsaacLabInfrastructure/source/ace_isaac_5/ace_isaac_5/mars_terrain/rocks_merged.usd",
        )
        self.valid_spawns = self.terrainManager.spawn_locations
        #self.target_spawns = torch.zeros(self.cfg.scene.num_envs, 3).cuda()

        # Goal Marker Initialization
        sphere_cfg = SPHERE_MARKER_CFG.copy()
        sphere_cfg.prim_path = "/Visuals/Command/position_goal"
        self.sphere_marker = VisualizationMarkers(cfg=sphere_cfg)
        self.sphere_marker.set_visibility(True)
        self.marker_radius = 7.0
        self.target_spawns = torch.zeros(self.cfg.scene.num_envs, 3, device="cuda:0")

        # self.target_position = torch.zeros(self.cfg.scene.num_envs, 3).cuda()

    def _visualize_markers(self):
        pass

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.robot.set_joint_velocity_target(self.actions, joint_ids=self.dof_idx)

    def _get_observations(self) -> dict:
        
        camera_data = self.robot_camera.data.output["rgb"] / 255.0
        # normalize the camera data for better training results
        mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
        camera_data -= mean_tensor

        return {"policy": camera_data.clone()}

    def _get_rewards(self) -> torch.Tensor:

        total_reward = torch.linalg.norm(self.robot.data.root_com_lin_vel_b, dim=-1, keepdim=True)
        return total_reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return False, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # default_root_state = self.robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # self.robot.write_root_state_to_sim(default_root_state, env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = self.valid_spawns[env_ids]
        default_root_state[:, 2] += 0.065
        self.robot.write_root_state_to_sim(default_root_state, env_ids)



        device="cuda:0"
        root_pos = default_root_state[:, :3]  # Shape [N_envs, 3]
        
        # 3) initialize a candidate tensor of targets
        N = len(env_ids)
        targets = torch.zeros_like(root_pos, device=device)


        # 4) first random draw on the circle
        angles = torch.rand(N, device=device) * 2 * math.pi
        targets[:, 0] = root_pos[:, 0] + self.marker_radius * torch.cos(angles)
        targets[:, 1] = root_pos[:, 1] + self.marker_radius * torch.sin(angles)
        targets[:, 2] = root_pos[:, 2]
        env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=device)

        while True:
            bad_envs, num_bad = self.terrainManager.check_if_target_is_valid(
                env_ids_tensor, targets, device=device
            )

            if num_bad == 0:
                break

            # build a mask of which rows to re-sample
            # for each row i, True if env_ids_tensor[i] is in bad_envs
            mask = (env_ids_tensor.unsqueeze(1) == bad_envs.unsqueeze(0)).any(dim=1)

            # re-sample angles for only the bad ones
            n_bad = mask.sum()
            new_angles = torch.rand(n_bad, device=device) * 2 * math.pi

            # re-generate those rows
            row_idx = mask.nonzero(as_tuple=True)[0]
            targets[row_idx, 0] = root_pos[row_idx, 0] + self.marker_radius * torch.cos(new_angles)
            targets[row_idx, 1] = root_pos[row_idx, 1] + self.marker_radius * torch.sin(new_angles)
            targets[row_idx, 2] = root_pos[row_idx, 2]
            # loop again until no bad targets remain

        # 6) write the fully validated targets back into your env-wide tensor
        #    (torch indexing with a Python list works too)
        self.target_spawns[env_ids] = targets
