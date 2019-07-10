from gym.envs.flex.flex_env import FlexEnv
# ^^^^^ so that user gets the correct error
# message if pydart is not installed correctly

from gym.envs.flex.granular_sweep_env import GranularSweepEnv
from gym.envs.flex.goo_sweep_env import GooSweepEnv
from gym.envs.flex.granular_sweep_voxel_bar_env import GranularSweepVoxelBarEnv
from gym.envs.flex.granular_sweep_torque_voxel_bar_env import GranularSweepTorqueVoxelBarEnv
from gym.envs.flex.granular_sweep_ghost_bar_env import GranularSweepGhostBarEnv
from gym.envs.flex.granular_sweep_ghost_bar_linear_control_env import GranularSweepGhostBarLinearControlEnv
from gym.envs.flex.granular_sweep_ghost_bar_raw_img_env import GranularSweepGhostBarRawImgEnv
from gym.envs.flex.granular_sweep_rot_base_env import GranularSweepRotBaseEnv