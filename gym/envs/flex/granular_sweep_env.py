import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from gym.envs.flex import flex_env
from os import path
import gym
import six
#
try:
    import bindings as pyFlex
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: PyFlex Binding is not installed correctly)".format(e))


class GranularSweepEnv(flex_env.FlexEnv):
    def __init__(self):
        obs_size = 32*32*2+4
        self.frame_skip = 5
        flex_env.FlexEnv.__init__(self,self.frame_skip,obs_size,np.array([[-2,-2,-1,-1],[2,2,1,1]]),scene=0)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

    def _step(self, action):

        done = self.do_simulation(action,self.frame_skip)
        obs = self._get_obs()
        reward = 0
        info = {}
        return obs,reward,done,info

    def _get_obs(self):
        state = flex_env.FlexEnv._get_obs(self)
        return state


if __name__ == '__main__':
    env = GranularSweepEnv()


    while True:
        env.reset()
        for _ in range(1000):
            # print(pyFlex.get_state())
            act = np.random.uniform([-2, -2, -1, -1]*env.numInstances, [2, 2, 1, 1]*env.numInstances,env.numInstances*4)
            obs,rwd,done,info = env.step(act)
            if done:
                break
        else:
            continue
        break
