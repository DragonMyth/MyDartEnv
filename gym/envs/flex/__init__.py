from gym.envs.flex.flex_env import FlexEnv
# ^^^^^ so that user gets the correct error
# message if pydart is not installed correctly

from gym.envs.flex.granular_sweep_env import GranularSweepEnv

from gym.envs.flex.granular_sweep_raw_img_env import GranularSweepRawImgEnv
from gym.envs.flex.granular_sweep_raw_img_ghost_control_env import GranularSweepRawImgGhostControlEnv
from gym.envs.flex.granular_sweep_raw_img_rwd_mean_std_env import GranularSweepRawImgRwdMeanStdEnv
from gym.envs.flex.granular_sweep_detailed_img_ghost_control_env import GranularSweepDetailedImgGhostControlEnv
from gym.envs.flex.plastic_reshaping_env import PlasticReshapingEnv
from gym.envs.flex.plastic_spring_reshaping_env import  PlasticSpringReshapingEnv
from gym.envs.flex.plastic_spring_multi_goal_env import PlasticSpringMultiGoalReshapingEnv
from gym.envs.flex.plastic_spring_multi_goal_knn_rwd_env import PlasticSpringMultiGoalKNNRWDEnv
from gym.envs.flex.plastic_spring_multi_goal_solid_env import PlasticSpringMultiGoalReshapingSolidEnv
from gym.envs.flex.plastic_test_env import PlasticTestEnv