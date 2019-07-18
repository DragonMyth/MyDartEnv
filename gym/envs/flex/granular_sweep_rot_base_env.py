import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from gym.envs.flex import flex_env

try:
    import bindings as pyFlex
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: PyFlex Binding is not installed correctly)".format(e))


class GranularSweepRotBaseEnv(flex_env.FlexEnv):
    def __init__(self):

        self.resolution = 11
        obs_size = self.resolution * self.resolution*2 + 12

        self.frame_skip = 6
        action_bound = np.array([[-4, -4, -4, -4,-np.pi/6], [4, 4, 4, 4,np.pi/6]])
        obs_high = np.ones(obs_size) * np.inf
        obs_low = -obs_high
        observation_bound = np.array([obs_low, obs_high])
        flex_env.FlexEnv.__init__(self, self.frame_skip, obs_size, observation_bound, action_bound, scene=4)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.action_scale = (action_bound[1] - action_bound[0]) / 2
        # self.circle_center = np.random.uniform(-2, 2, (self.numInstances, 2))
        # self.center_list = np.array([[1.5,1.5],[-1.5,-1.5],[-1.5,1.5],[1.5,-1.5]])

        # self.center_list = np.array([[0,1.5],[0,-1.5]])
        # self.center_list = np.array([[0,0]])


        self.center_list = np.random.uniform(-2,2,(100,2))

        self.randGoalRange = self.center_list.shape[0]

        self.circle_center = np.random.random_integers(0,self.randGoalRange,self.numInstances)


        self.goal_gradients = np.zeros((self.numInstances,self.resolution,self.resolution))
        self.iter_num = 5000

    def _step(self, action):
        action = action * self.action_scale
        prev_state = self.get_state()
        centers = self.center_list[self.circle_center]




        prev_rot = prev_state[:,2]
        transformed_centers = self.get_transformed_goal(centers,prev_rot)

        expanded_centers_prev = np.expand_dims(transformed_centers, axis=1)
        expanded_centers_prev = np.repeat(expanded_centers_prev, prev_state.shape[1], axis=1)

        prev_distance = 0.1*np.sum(np.linalg.norm(prev_state - expanded_centers_prev, axis=2)[:, 6::]**3, axis=1)

        action = np.concatenate([action,transformed_centers],axis=1)
        done = self.do_simulation(action, self.frame_skip)


        curr_state = self.get_state()

        curr_rot = curr_state[:,2]

        transformed_centers = self.get_transformed_goal(centers,curr_rot)

        expanded_centers_curr = np.expand_dims(transformed_centers, axis=1)
        expanded_centers_curr = np.repeat(expanded_centers_curr, prev_state.shape[1], axis=1)

        curr_distance = 0.1*np.sum(np.linalg.norm(curr_state - expanded_centers_curr, axis=2)[:, 6::]**3, axis=1)

        energy = np.clip(np.sum(np.linalg.norm(prev_state[:, :2] - curr_state[:, :2], axis=1),axis=1),0,0.02)


        rewards = prev_distance- curr_distance
        old_rwd = rewards
        rewards[rewards<0.02]=energy[rewards<0.02]

        info = {'Total Reward': rewards[0],'Distance Reward':old_rwd[0]}
        obs = self._get_obs()

        return obs, rewards, done, info

    def get_transformed_goal(self,goal,rot):

        # rot[:,0] = 0
        # rot[:,1] = 1
        new_prev_cent_x = goal[:, 0] * rot[:, 0] + goal[:, 1] * rot[:, 1]
        new_prev_cent_x = np.expand_dims(new_prev_cent_x,1)

        new_prev_cent_y = -goal[:, 0] * rot[:, 1] + goal[:, 1] * rot[:, 0]
        new_prev_cent_y = np.expand_dims(new_prev_cent_y,1)


        return np.concatenate([new_prev_cent_x, new_prev_cent_y], axis=1)
    def _get_obs(self):

        states = self.get_state()
        obs_list = []

        transformed_goals = self.get_transformed_goal(self.center_list[self.circle_center],states[:,2])
        for i in range(self.numInstances):
            state = states[i]
            part_state = state[6::]
            bar_state = state[:6]
            bar_density = self.get_voxel_bar_density(bar_state)
            density = self.get_particle_density(part_state, normalized=True)

            goal_gradient = self.get_goal_gradient(transformed_goals[i])

            bar_pos = bar_state[0:2].flatten()
            bar_vel = bar_state[3:5].flatten()
            base_pos = bar_state[2].flatten()
            base_angVel = np.array([np.cos(bar_state[5,0]),np.cos(bar_state[5,1])]).flatten()

            bar_info = np.concatenate([bar_pos,base_pos,bar_vel,base_angVel])
            obs = np.concatenate([bar_info.flatten(),density.flatten()-goal_gradient.flatten(),bar_density.flatten()])

            obs_list.append(obs)

        return np.array(obs_list)

    def get_goal_gradient(self,goal):
        x, y = np.meshgrid(np.linspace(-4, 4, self.resolution), np.linspace(-4, 4, self.resolution))
        sigma = 0.7

        gradient = np.exp(-(((x - goal[0]) ** 2+(y-goal[1])**2) / (2.0 * sigma ** 2)))
        return gradient
    def get_voxel_bar_density(self,bar_state):
        centerBar1 = bar_state[0]
        centerBar2 = bar_state[1]


        directionBar1 = np.array([0,1])


        directionBar2 = np.array([1,0])

        ## length of bar is 0.7, half length is 0.35
        end_point_bar1_1 = centerBar1+directionBar1*0.7
        end_point_bar1_2 = centerBar1-directionBar1*0.7

        end_point_bar2_1 = centerBar2+directionBar2*0.7
        end_point_bar2_2 = centerBar2-directionBar2*0.7

        step = 1.0/40
        interp = np.arange(0,1+step,step)
        interp = np.expand_dims(interp, axis=1)
        interp = np.repeat(interp, 2, axis=1)
        interp_bar1 = (1-interp)*end_point_bar1_1+interp*end_point_bar1_2
        interp_bar2 = (1-interp)*end_point_bar2_1+interp*end_point_bar2_2

        interp_bar1_x = interp_bar1[:,0]
        interp_bar1_y = interp_bar1[:,1]


        interp_bar2_x = interp_bar2[:,0]
        interp_bar2_y = interp_bar2[:,1]
        interp_y = np.concatenate([interp_bar1_y,interp_bar2_y])
        interp_x = np.concatenate([interp_bar1_x,interp_bar2_x])

        H, xedges, yedges = np.histogram2d(interp_y, interp_x, bins=[self.resolution, self.resolution],
                                           range=[[-4, 4], [-4, 4]])
        H = (H>0).astype(int)




        return H
    def get_particle_density(self, particles, normalized=True):
        x_pos = particles[:, 0]
        y_pos = particles[:, 1]

        H, xedges, yedges = np.histogram2d(y_pos, x_pos, bins=[self.resolution, self.resolution],
                                           range=[[-4, 4], [-4, 4]])

        # H = np.clip(H,0,1000)

        if normalized:
            H = H / particles.shape[0]
            H = H ** (1.0 / 3)

        return H

    def _reset(self):
        # self._seed(self.seed)
        flex_env.FlexEnv._reset(self)
        # if (self.iter_num < threshold):
        #     curriculum = 0
        # else:
        #     curriculum = 1 - np.exp(-0.005 * (self.iter_num - threshold))
        self.iter_num += 1
        self.circle_center = np.random.random_integers(0,self.randGoalRange,self.numInstances)

        for i in range(self.numInstances):
            self.goal_gradients[i] = self.get_goal_gradient(self.center_list[self.circle_center[i]])
        # self.circle_center = np.ones((self.numInstances, 2)) * 1.5
        # self.circle_center = np.zeros((self.numInstances, 2)) + np.random.uniform(-2, 2,
        #                                                                           (self.numInstances, 2)) * curriculum
        # self.circle_center
        # print("Target: ", self.circle_center)
        # print("Curriculum: ",curriculum)
        return self._get_obs()


if __name__ == '__main__':
    env = GranularSweepRotBaseEnv()
    #
    # while True:
    #     env.reset()
    #     for _ in range(1000):
    #         # print(pyFlex.get_state())
    #         # act = np.random.uniform([-4, -4, -1, -1], [4, 4, 1, 1],(25,4))
    #         act = np.zeros((25, 4))
    #
    #         obs, rwd, done, info = env.step(act)
    #         if done:
    #             break
    #     else:
    #         continue
    #     break
    #env.save_video = True
    # while True:
    env.reset()
    for _ in range(1000):
        # print(pyFlex.get_state())
        # act = np.random.uniform([-4, -4, -1, -1], [4, 4, 1, 1],(25,4))
        act = np.zeros((25, 5))
        act[:,4]=1
        obs, rwd, done, info = env.step(act)
        if done:
            break
    # else:
    #     continue
    # break