# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import TiledCamera
import isaaclab.utils.math as math_utils

from .jackal_terrain_env_cfg import JackalTerrainEnvCfg
from .terrain_utils.terrain_utils import TerrainManager
from .terrain_utils.usd_utils import get_triangles_and_vertices_from_prim

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


class JackalTerrainEnv(DirectRLEnv):
    cfg: JackalTerrainEnvCfg

    def __init__(self, cfg: JackalTerrainEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.dof_idx, _ = self.robot.find_joints(self.cfg.dof_names)

    def get_spawns(self, num_envs: int, dist: float):
        """
        Build self.valid_spawns2 by greedy spacing on (x, y):
        - start with the first point in self.valid_spawns
        - add the next point whose (x,y) distance is >= dist from all selected
        - stop when reaching num_envs or no more valid points
        Returns: self.valid_spawns2 (K, D), where K <= min(num_envs, N)
        """
        pts = self.valid_spawns                 # (N, D), e.g., D=3 for (x,y,z)
        if pts.numel() == 0:
            valid_spawns2 = pts.new_zeros((0, pts.shape[1]))
            return valid_spawns2

        N, D = pts.shape
        target_count = min(num_envs, N)
        xy = pts[:, :2]                         # only (x, y) for distance checks
        dist2 = float(dist) * float(dist)

        # Always include the first point
        selected_idx = [0]

        # Track minimum squared distance to the selected set for each point
        min_d2 = torch.sum((xy - xy[0])**2, dim=1)

        # Candidates are those at least 'dist' from ALL selected so far
        valid = (min_d2 >= dist2)
        valid[0] = False  # already selected

        while len(selected_idx) < target_count:
            cand = torch.nonzero(valid, as_tuple=False).squeeze(1)
            if cand.numel() == 0:
                break  # no more points satisfy the spacing

            # Take the next valid point in original order (first index)
            i = int(cand[0].item())
            selected_idx.append(i)

            # Update min distance-to-selected with the new point
            d2 = torch.sum((xy - xy[i])**2, dim=1)
            min_d2 = torch.minimum(min_d2, d2)

            # Recompute validity against all selected (via min distance)
            valid = (min_d2 >= dist2)
            # ensure already-selected don't get re-picked
            valid[torch.tensor(selected_idx, device=pts.device)] = False

        valid_spawns2 = pts[torch.tensor(selected_idx, device=pts.device, dtype=torch.long)]
        return valid_spawns2

    def _setup_scene(self):

        self.gpu = "cuda:0"

        self.robot = Articulation(self.cfg.robot_cfg)
        self.robot_camera = TiledCamera(self.cfg.tiled_camera)

        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["robot"] = self.robot
        self.scene.sensors["tiled_camera"] = self.robot_camera 
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
        self.goal_radii = torch.empty(self.scene.num_envs, device=self.gpu, dtype=torch.float32)

        # Data structure to store observation history
        self.history_len = 5
        self._camera_hist: torch.Tensor | None = None

        # Terrain Manager
        self.terrainManager = TerrainManager(
            num_envs=self.cfg.scene.num_envs, 
            device=self.gpu,
        )
        self.valid_spawns = self.terrainManager.spawn_locations
        self.valid_spawns = self.get_spawns(self.cfg.scene.num_envs, 25.0)

        # Get Mesh vertices and faces
        terrain_prim_path = "/World/terrain/terrain/ground"
        faces, self.mesh_vertices = get_triangles_and_vertices_from_prim(terrain_prim_path)
        self.mesh_vertices = torch.from_numpy(self.mesh_vertices).to(self.gpu)
        self.face_vertices = self.mesh_vertices[faces]

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
        self.scene.write_data_to_sim()

    def _get_observations(self) -> dict:

        self.scene.update(1.0/120.0)

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

        N, T, H, W, C = self._camera_hist.shape

        #bodyvel = self.robot.data.root_com_lin_vel_b
        goal_vec = self.target_spawns - self.robot.data.root_pos_w
        goal_dist = torch.linalg.norm(goal_vec, dim=-1, keepdim=True)
        unit_goal = self._get_goal_vec_normalized()
        state_input = torch.hstack((unit_goal, goal_dist))
        _, S = state_input.shape
        self.state_input = state_input[:, None, None, None, :].expand(N, T, H, W, S)

        final_obs = torch.cat([self._camera_hist.clone(), self.state_input.clone()], dim=-1) # (N,T,H,W,C+4)

        # if (self.common_step_counter % 100 == 0 and self.common_step_counter != 0):
        #     import pdb; pdb.set_trace()

        return {"policy": final_obs}
    
    def _get_rewards(self) -> torch.Tensor:

        # Angle Alignment Reward
        goal_vec = self.target_spawns - self.robot.data.root_pos_w 
        goal_vec[:,2] = 0.0
        goal_vec = goal_vec/torch.linalg.norm(goal_vec, dim=-1, keepdim=True)
        forwards = math_utils.quat_apply(self.robot.data.root_link_quat_w, self.robot.data.FORWARD_VEC_B)
        forwards[:,2] = 0.0
        forwards = forwards/torch.linalg.norm(forwards, dim=-1, keepdim=True)
        alignment = torch.sum(forwards * goal_vec, dim=-1, keepdim=True)
        angle = torch.acos(alignment.clamp(-1.0, 1.0))
        vel = self.robot.data.root_com_lin_vel_b[:,0].reshape(-1,1) # torch.Size([N, 1])
        threshold = math.pi / 9.0
        mask = (angle <= threshold).to(torch.float32) # torch.Size([N, 1])
        forward_reward = vel * mask # torch.Size([N, 1])
        alignment_reward = forward_reward*torch.exp(alignment) # torch.Size([N, 1])

        # Arrival Bonus
        vec3D = self.target_spawns - self.robot.data.root_pos_w
        dist2D = torch.linalg.norm(vec3D[:, :2], dim=-1)
        arrived_mask = dist2D < 0.5
        arrival_bonus = arrived_mask.to(torch.float32)*2.0 

        # Distance Penalty 
        dist3D = torch.linalg.norm(vec3D, dim=-1)
        dist_penalty = -0.1 * dist3D

        total_reward = alignment_reward + arrival_bonus + dist_penalty                                 # torch.Size([N, 1])

        return total_reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        time_out = self.episode_length_buf >= (self.max_episode_length - 1)

        dist = self.target_spawns - self.robot.data.root_pos_w 
        dist_xy = torch.linalg.norm(dist[:, :2], dim=-1)

        reached = dist_xy < 0.5

        # if reached:
        #     print("GOAL REACHED: TIMEOUT HERE")

        return reached, time_out
    

    @torch.no_grad()    
    def _nearest_vertex_heights_from_xy(self, targets_xy: torch.Tensor,
                                        verts_xyz: torch.Tensor,
                                        chunk_size: int = 100_000) -> torch.Tensor:
        """
        For each (x,y) in targets_xy, find the closest vertex in verts_xyz (by L2 in XY)
        and return its z. All tensors must be on the same device.
        """
        device = targets_xy.device
        verts_xy = verts_xyz[:, :2]                               # (M, 2)
        N = targets_xy.shape[0]
        M = verts_xy.shape[0]

        best_d2  = torch.full((N,), float("inf"), device=device)
        best_idx = torch.full((N,), -1, dtype=torch.long, device=device)

        offset = 0
        while offset < M:
            end = min(offset + chunk_size, M)
            chunk = verts_xy[offset:end]                          # (C, 2)

            # (N, C) squared distances without sqrt for speed
            diff  = targets_xy[:, None, :] - chunk[None, :, :]    # (N, C, 2)
            d2    = (diff * diff).sum(dim=2)                      # (N, C)

            # best in this chunk
            min_d2, min_j = torch.min(d2, dim=1)                  # (N,), (N,)

            # keep global best
            better = min_d2 < best_d2
            best_d2[better]  = min_d2[better]
            best_idx[better] = min_j[better] + offset

            offset = end

        return verts_xyz[best_idx, 2]                              # (N,)
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] = self.valid_spawns[env_ids]
        default_root_state[:, 2] += 0.075
        self.robot.write_root_state_to_sim(default_root_state, env_ids)

        angles = torch.empty(len(env_ids), device=self.gpu).uniform_(-math.pi/8.0, math.pi/8.0)
        self.goal_radii[env_ids] = self.goal_radii[env_ids].uniform_(4.0, 6.0)
        targets = default_root_state[:, :3].clone()
        targets[:, 0] = targets[:, 0] + self.goal_radii[env_ids] * torch.cos(angles)
        targets[:, 1] = targets[:, 1] + self.goal_radii[env_ids] * torch.sin(angles)
        targets_xy = targets[:, :2]
        targets[:, 2] = self._nearest_vertex_heights_from_xy(targets_xy, self.mesh_vertices)
        targets[:, 2] += 0.5

        self.target_spawns[env_ids] = targets
        self._visualize_markers()
        self._visualize_markers()
