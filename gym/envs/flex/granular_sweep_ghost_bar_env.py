import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from gym.envs.flex import flex_env

try:
    import bindings as pyFlex
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: PyFlex Binding is not installed correctly)".format(e))


class GranularSweepGhostBarEnv(flex_env.FlexEnv):
    def __init__(self):
        self.resolution = 64
        obs_size = self.resolution * self.resolution*3 + 10

        self.frame_skip = 3
        action_bound = np.array([[-4, -4, -1, -1], [4, 4, 1, 1]])
        obs_high = np.ones(obs_size) * np.inf
        obs_low = -obs_high
        observation_bound = np.array([obs_low, obs_high])
        flex_env.FlexEnv.__init__(self, self.frame_skip, obs_size, observation_bound, action_bound, scene=3)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.action_scale = (action_bound[1] - action_bound[0]) / 2
        # self.circle_center = np.random.uniform(-2, 2, (self.numInstances, 2))

        self.circle_center = np.random.random_integers(0,0,self.numInstances)
        # self.center_list = np.array([[1.5,1.5],[-1.5,-1.5],[-1.5,1.5],[1.5,-1.5]])
        self.center_list = np.array([[0,1.5]])
        self.goal_gradients = np.zeros((self.numInstances,self.resolution,self.resolution))
        self.iter_num = 5000

    def _step(self, action):
        action = action * self.action_scale
        prev_state = self.get_state()
        centers = self.center_list[self.circle_center]

        expanded_centers = np.expand_dims(centers, axis=1)
        expanded_centers = np.repeat(expanded_centers, prev_state.shape[1], axis=1)

        prev_distance = 0.1*np.sum(np.linalg.norm(prev_state - expanded_centers, axis=2)[:, 4::]**2, axis=1)

        action = np.concatenate([action,centers],axis=1)
        done = self.do_simulation(action, self.frame_skip)

        curr_state = self.get_state()
        curr_distance = 0.1*np.sum(np.linalg.norm(curr_state - expanded_centers, axis=2)[:, 4::]**2, axis=1)

        rewards = prev_distance- curr_distance
        # prev_obs = self._get_obs()
        # densities = prev_obs[:, 10:10 + self.resolution * self.resolution]
        # goals = prev_obs[:, -self.resolution * self.resolution::]
        #
        # prev_rewards =  np.sum(densities * goals, axis=1)
        # done = self.do_simulation(action, self.frame_skip)
        #
        info = {}
        obs = self._get_obs()
        # densities = obs[:,10:10+self.resolution*self.resolution]
        # goals = obs[:,-self.resolution*self.resolution::]
        #
        # rewards = np.sum(densities*goals,axis=1)-prev_rewards
        return obs, rewards, done, info

    def _get_obs(self):

        states = self.get_state()
        obs_list = []
        for i in range(self.numInstances):
            state = states[i]
            part_state = state[4::]
            bar_state = state[:4]
            bar_density = self.get_voxel_bar_density(bar_state)
            density = self.get_particle_density(part_state, normalized=True)
            goal_gradient = self.get_goal_gradient(self.center_list[self.circle_center[i]])

            obs = np.concatenate([bar_state.flatten(), self.center_list[self.circle_center[i]],density.flatten()-goal_gradient.flatten(),bar_density.flatten(),goal_gradient.flatten()])

            obs_list.append(obs)

        return np.array(obs_list)

    def get_goal_gradient(self,goal):
        x, y = np.meshgrid(np.linspace(-4, 4, self.resolution), np.linspace(-4, 4, self.resolution))
        sigma = 0.5

        gradient = np.exp(-(((x - goal[0]) ** 2+(y-goal[1])**2) / (2.0 * sigma ** 2)))
        return gradient
    def get_voxel_bar_density(self,bar_state):
        center = bar_state[0]
        direction = bar_state[1]
        ## length of bar is 0.7, half length is 0.35
        end_point_1 = center+direction*0.7
        end_point_2 = center-direction*0.7

        step  = 1.0/40
        interp = np.arange(0,1+step,step)
        interp = np.expand_dims(interp, axis=1)
        interp = np.repeat(interp, 2, axis=1)
        interp = (1-interp)*end_point_1+interp*end_point_2
        interp_x = interp[:,0]
        interp_y = interp[:,1]

        H, xedges, yedges = np.histogram2d(interp_y, interp_x, bins=[self.resolution, self.resolution],
                                           range=[[-4, 4], [-4, 4]])
        H = np.ceil(H/np.max(H))

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
        threshold = 50
        # if (self.iter_num < threshold):
        #     curriculum = 0
        # else:
        #     curriculum = 1 - np.exp(-0.005 * (self.iter_num - threshold))
        self.iter_num += 1
        self.circle_center = np.random.random_integers(0,0,self.numInstances)
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
    env = GranularSweepGhostBarEnv()
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
        act = np.zeros((25, 4))

        obs, rwd, done, info = env.step(act)
        if done:
            break
    # else:
    #     continue
    # break
