import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from gym.envs.flex import flex_env

try:
    import bindings as pyFlex
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: PyFlex Binding is not installed correctly)".format(e))


class GooSweepEnv(flex_env.FlexEnv):
    def __init__(self):
        self.resolution = 32
        obs_size = self.resolution*self.resolution+8

        self.frame_skip = 5
        action_bound = np.array([[-1,-1,-1,-1],[1,1,1,1]])
        obs_high = np.ones(obs_size)*np.inf
        obs_low = -obs_high
        observation_bound = np.array([obs_low,obs_high])
        flex_env.FlexEnv.__init__(self,self.frame_skip,obs_size,observation_bound,action_bound,scene=1)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.action_scale = (action_bound[1]-action_bound[0])/2
    def _step(self, action):
        action = action*self.action_scale
        prev_state = self.get_state()
        prev_distance = np.sum(np.linalg.norm(prev_state, axis=2)[:,4::], axis=1)

        done = self.do_simulation(action,self.frame_skip)
        curr_state = self.get_state()
        curr_distance = np.sum(np.linalg.norm(curr_state, axis=2)[:,4::], axis=1)

        rewards = prev_distance-curr_distance
        # print(reward.mean())
        info = {}
        obs = self._get_obs()
        return obs,rewards,done,info

    def _get_obs(self):

        states = self.get_state()
        obs_list = []
        for i in range(self.numInstances):
            state = states[i]
            part_state = state[4::]
            bar_state = state[:4]
            density = self.get_particle_density(part_state)
            obs = np.concatenate([bar_state.flatten(),density.flatten()])
            obs_list.append(obs)

        return np.array(obs_list)

    def get_particle_density(self,particles,normalized=True):
        x_pos = particles[:,0]
        y_pos = particles[:,1]

        H,xedges,yedges = np.histogram2d(y_pos,x_pos,bins=[self.resolution,self.resolution],range=[[-1,1],[-1,1]])

        H = np.clip(H,0,2000)

        if normalized:
            H = H/2000.0

        return H

    def _reset(self):
        # self._seed(self.seed)
        flex_env.FlexEnv._reset(self)
        return self._get_obs()
if __name__ == '__main__':
    env = GooSweepEnv()


    while True:
        env.reset()
        for _ in range(1000):
            # print(pyFlex.get_state())
            # act = np.random.uniform([-4, -4, -1, -1], [4, 4, 1, 1],(25,4))
            act =np.zeros((25,4))

            obs,rwd,done,info = env.step(act)
            if done:
                break
        else:
            continue
        break
