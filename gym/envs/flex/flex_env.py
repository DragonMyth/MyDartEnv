import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six
import pygame as pg
try:
    import bindings as pyFlex
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: PyFlex Binding is not installed correctly)".format(e))


class FlexEnv(gym.Env):
    def __init__(self, frame_skip, observation_size, observation_bounds,action_bounds,
                 dt=1 / 60.0, obs_type="parameter", action_type="continuous", scene=0, disableViewer = True):
        assert obs_type in ('parameter', 'image')
        assert action_type in ("continuous", "discrete")

        self.disableViewer = disableViewer
        self.screen = None
        self.screen_size = (400,400)
        self.sub_screen_gap = 4
        if not self.disableViewer:
            pg.init()
            self.screen = pg.display.set_mode(self.screen_size,display=pg.OPENGL)

        pyFlex.chooseScene(scene)
        pyFlex.initialize()

        pyFlex.setDt(dt)
        self.dt = dt
        print('pyFlex initialization OK')

        self._obs_type = obs_type
        self.frame_skip = frame_skip
        self.numInstances = pyFlex.getNumInstances()

        # assert not done
        self.obs_dim = observation_size
        self.act_dim = len(action_bounds[0])
        # for discrete instances, action_space should be defined in the subclass
        self.action_space = spaces.Box(action_bounds[0], action_bounds[1])
        self.observation_space = spaces.Box(observation_bounds[0], observation_bounds[1])

        self.metadata = {
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.save_video = False

        self.video_path = "/home/yzhang/data"
        self.step_cnt = 0

    def _seed(self, seed=None):
        if seed is None:
            pyFlex.setSceneRandSeed(-1)
        else:
            np.random.seed(seed)
            pyFlex.setSceneRandSeed(seed)
        # self.np_random, sseed = seeding.np_random(seed)

        return [seed]

    def do_simulation(self, tau, n_frames):
        self.step_cnt += 1

        done = False
        save = self.save_video
        for _ in range(n_frames):
            path = os.path.join(self.video_path, 'frame_%d.tga' % self.step_cnt)
            pyFlex.update_frame(save,path,tau.flatten())
            save = False
            done=pyFlex.sdl_main()
            if done:
                break

        return done
    def _step(self, action):
        done = self.do_simulation(action,self.frame_skip)
        obs = self.get_state()
        reward = 0
        info = {}
        return obs,reward,done,info


    def _reset(self):
        # self._seed(self.seed)
        return pyFlex.resetScene()

    def get_state(self):
        state_vec = pyFlex.get_state()
        # part_state = state_vec[:-4*self.numInstances]
        # bar_state = state_vec[-4*self.numInstances::]
        #
        # part_state = part_state.reshape([self.numInstances,int(part_state.shape[0]/self.numInstances),2])
        # bar_state = bar_state.reshape([self.numInstances,int(bar_state.shape[0]/self.numInstances),2])
        #
        # full_state = np.concatenate([bar_state,part_state],axis=1)
        # return full_state
        full_state = state_vec.reshape([self.numInstances,int(state_vec.shape[0]/self.numInstances),3])
        return full_state

    def _render(self, mode='human', close=False):
        if(self.disableViewer):
            return
        else:
            pg.display.update()
            img = pg.surfarray.array3d(self.screen).transpose([1,0,2])
            return img
if __name__ == '__main__':
    env = FlexEnv(frame_skip=5, observation_size=10, action_bounds=np.array([[-3, -3, -1, -1], [3, 3, 1, 1]]),scene=0)

    while True:
        pyFlex.resetScene()
        for _ in range(5000):
            # print(pyFlex.get_state())
            act = np.random.uniform([-2, -2, -1, -1]*env.numInstances, [2, 2, 1, 1]*env.numInstances,env.numInstances*4)
            obs,rwd,done,info = env.step(act)
            if done:
                break
        else:
            continue
        break
