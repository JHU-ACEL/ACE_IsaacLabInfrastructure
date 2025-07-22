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




@configclass
class MarsTerrainSceneCfg(InteractiveSceneCfg):
    """
    Mars Terrain Scene Configuration
    """
    # Hidden Terrain (merged terrain of ground and obstacles) for raycaster.
    # This is done because the raycaster doesn't work with multiple meshes
    # hidden_terrain = AssetBaseCfg(
    #     prim_path="/World/terrain/hidden_terrain",
    #     spawn=sim_utils.UsdFileCfg(
    #         visible=False,
    #         usd_path="/home/bchien1/ACE_IsaacLabInfrastructure/source/ace_isaac_5/ace_isaac_5/mars_terrain/terrain_merged.usd",
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    # )



    # Obstacles
    # obstacles = AssetBaseCfg(
    #     prim_path="/World/terrain/obstacles",
    #     spawn=sim_utils.UsdFileCfg(
    #         visible=True,
    #         usd_path="/home/bchien1/ACE_IsaacLabInfrastructure/source/ace_isaac_5/ace_isaac_5/mars_terrain/rocks_merged.usd",
    #     ),
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    # )

        # Ground Terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/terrain",
        terrain_type="usd",
        usd_path="/home/bchien1/ACE_IsaacLabInfrastructure/source/ace_isaac_5/ace_isaac_5/mars_terrain/terrain_only.usd",
    )


@configclass
class JackalEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 7.0
    # - spaces definition
    action_space = 4
    # observation_space = 9
    #observation_space = 3
    state_space = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    # robot(s)
    robot_cfg: ArticulationCfg = JACKAL_CONFIG.replace(prim_path="/World/envs/env_.*/Robot")

    # camera
    # tiled_camera: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Robot/chassis/rgb_camera/jetbot_camera",
    #     #offset=TiledCameraCfg.OffsetCfg(pos=(-5.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    #     data_types=["rgb"],
    #     spawn=None,
    #     width=224,
    #     height=224,
    # )

    #goal_cfg = RigidObjectCfg(prim_path="/World/envs/env_.*/marker", spawn=sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/red_block.usd", scale = (3.0, 3.0, 3.0)), init_state=RigidObjectCfg.InitialStateCfg(pos=(2.1,-.3,0)))

    #observation_space = [5, tiled_camera.height, tiled_camera.width, 3]
    #observation_space = [tiled_camera.height, tiled_camera.width, 3]


    observation_space = 3

    # scene
    scene: MarsTerrainSceneCfg = MarsTerrainSceneCfg(num_envs=1, env_spacing=20.0, replicate_physics=True)
    #scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=5.0, replicate_physics=True)
    dof_names = ['front_left_wheel_joint', 'front_right_wheel_joint', 'rear_left_wheel_joint', 'rear_right_wheel_joint']