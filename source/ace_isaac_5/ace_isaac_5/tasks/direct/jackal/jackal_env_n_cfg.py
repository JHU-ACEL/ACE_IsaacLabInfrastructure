# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from ace_isaac_5.robots.jackal import JACKAL_CONFIG

from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab.sensors import TiledCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab.terrains import TerrainImporterCfg
#from .terrain_utilities.terrain_importer import RoverTerrainImporter



@configclass
class JackalEnvNCfg(DirectRLEnvCfg):

    episode_length_s = 10.0

    # simulation
    decimation = 2
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot(s)
    robot_cfg: ArticulationCfg = JACKAL_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    
    # sensors
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_link/bumblebee_stereo_camera_frame/bumblebee_stereo_right_frame/bumblebee_stereo_right_camera",
        #offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=None,
        width=224,
        height=224,
    )


    # - spaces definition
    state_space = 0
    action_space = 4
    observation_space = [5, tiled_camera.height, tiled_camera.width, 3]
    #observation_space = [tiled_camera.height, tiled_camera.width, 3]

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=80, env_spacing=30.0, replicate_physics=True)


    goal_cfg = RigidObjectCfg(prim_path="/World/envs/env_.*/marker", spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd", scale = (6.0, 6.0, 10.0)))
    dof_names = ['front_left_wheel_joint', 'front_right_wheel_joint', 'rear_left_wheel_joint', 'rear_right_wheel_joint']