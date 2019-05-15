import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six

try:
    import bindings as pyFlex
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: PyFlex Binding is not installed correctly)".format(e))


class FlexEnv(gym.Env):
    def __init__(self, frame_skip, observation_size, action_bounds,
                 dt=1 / 60.0, obs_type="parameter", action_type="continuous", scene=0):
        assert obs_type in ('parameter', 'image')
        assert action_type in ("continuous", "discrete")
        pyFlex.chooseScene(scene)
        pyFlex.initialize()
        pyFlex.setDt(dt)
        self.dt = dt
        print('pyFlex initialization OK')

        self._obs_type = obs_type
        self.frame_skip = frame_skip
        self.numInstances = pyFlex.getNumInstances()

        # assert not done
        self.obs_dim = observation_size*self.numInstances
        self.act_dim = len(action_bounds[0])**self.numInstances
        # for discrete instances, action_space should be defined in the subclass
        if action_type == "continuous":
            self.action_space = spaces.Box(action_bounds[1]*self.numInstances, action_bounds[0]*self.numInstances)

        # initialize the viewer, get the window size
        # initial here instead of in _render
        # in image learning
        # Give different observation space for different kind of envs
        # if self._obs_type == 'parameter':
        #     high = np.inf * np.ones(self.obs_dim)
        #     low = -high
        #     self.observation_space = spaces.Box(low, high)
        # elif self._obs_type == 'image':
        #     # Change to grayscale image later
        #     self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height))
        # else:
        #     raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

        self._seed()

        # self.viewer = None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

    def _seed(self, seed=None):
        if seed is None:
            pyFlex.setSceneRandSeed(-1)
        else:
            pyFlex.setSceneRandSeed(seed)
        self.np_random, seed = seeding.np_random(seed)

        # print(seed)

        return [seed]

    def do_simulation(self, tau, n_frames):
        done = False
        for _ in range(n_frames):
            pyFlex.update_frame(tau)

            done=pyFlex.sdl_main()
            if done:
                break
        return done
    def _step(self, action):

        done = self.do_simulation(action,self.frame_skip)
        obs = self._get_obs()
        reward = 0
        info = {}
        return obs,reward,done,info


    def _reset(self):
        return pyFlex.resetScene()

    def _get_obs(self):
        return pyFlex.get_state()

if __name__ == '__main__':
    env = FlexEnv(frame_skip=5, observation_size=10, action_bounds=np.array([[-3, -3, -1, -1], [3, 3, 1, 1]]),scene=0)


    main_loop_quit = False
    while True:
        pyFlex.resetScene()
        for _ in range(5000):
            # print(pyFlex.get_state())
            act = np.random.uniform([-2, -2, -1, -1]*env.numInstances, [2, 2, 1, 1]*env.numInstances,env.numInstances*4)
            env.step(act)
            main_loop_quit = pyFlex.sdl_main()
            if main_loop_quit:
                break
        else:
            continue
        break
