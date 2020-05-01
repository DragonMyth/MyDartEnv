import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six
import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
try:
    import bindings as pyFlex
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: PyFlex Binding is not installed correctly)".format(e))


class FlexEnv(gym.Env):
    def __init__(self, frame_skip, observation_size, observation_bounds,action_bounds,
                 dt=1 / 60.0, obs_type="parameter", action_type="continuous", scene=0, viewer=1):
        assert obs_type in ('parameter', 'image')
        assert action_type in ("continuous", "discrete")


        self.viewerId = viewer
        self.screen = None
        self.screen_size = (800,800)
        self.sub_screen_gap = 4

        flex_viewer = False
        if self.viewerId == 1:
            flex_viewer = True
        elif self.viewerId == 2:
            pg.init()
            # self.screen = pg.display.set_mode(self.screen_size)
            self.screen = pg.display.set_mode(self.screen_size,DOUBLEBUF|OPENGL)

            gluOrtho2D(0,self.screen_size[0],self.screen_size[1],0)
            # gluPerspective(45, (self.screen_size[0] / self.screen_size[1]), 0.1, 50.0)
        elif self.viewerId == 3:
            flex_viewer = True
            self.screen = pg.display.set_mode(self.screen_size)


        pyFlex.setVisualize(flex_viewer)
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
        # self.video_path = "/home/dragonmyth/data"

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
        path = os.path.join(self.video_path, 'frame_%d.tga' % self.step_cnt)

        pyFlex.simulateKSteps(save,path,tau.flatten(),n_frames)
        # for _ in range(n_frames):
        #     path = os.path.join(self.video_path, 'frame_%d.tga' % self.step_cnt)
        #     pyFlex.update_frame(save,path,tau.flatten())
        #     save = False
        #     done=pyFlex.sdl_main()
        #     if done:
        #         break

        return done
    def _step(self, action):
        done = self.do_simulation(action,self.frame_skip)
        obs = self.get_state()
        reward = 0
        info = {}
        return obs,reward,done,info


    def _reset(self):
        # print("ASFASFASFASF")
        # self._seed(self.seed)
        state = pyFlex.resetScene()
        palceholder_act = np.zeros((self.numInstances,7))
        palceholder_act[:,-1] = 1
        self.do_simulation(palceholder_act, 200)

        
        return state
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

    def set_controller(self,controllerConfig):
        pyFlex.setController(controllerConfig)

    def set_goal(self,goals):
        pyFlex.setGoal(goals)
    def setMapHalfExtent(self,halfExtent):
        pyFlex.setMapHalfExtent(halfExtent)
    def setInitClusterParam(self,clusterParam):
        pyFlex.setInitClusterParam(clusterParam)
    def get_density(self,particles,resolution,width,mapHalfExtent):
        return pyFlex.getParticleDensity(particles, resolution,width,mapHalfExtent)
    def get_height_map(self,particles,heights,resolution,width,mapHalfExtent):
        return pyFlex.getParticleHeightMap(particles,heights,resolution,width,mapHalfExtent)
    def get_angular_vel_flex(self,partPos,partVel):
        return pyFlex.getParticleAngularVelocity(partPos,partVel)
    def set_aux_info(self,auxInfo):
        return pyFlex.setAuxInfo(auxInfo)
    def _render(self, mode='human', close=False):
        if(self.viewerId==2):
            # pg.display.update()
            pg.display.flip()
            # img = pg.surfarray.array3d(self.screen).transpose([1, 0, 2])
            img = np.zeros((1,1,3))
        elif(self.viewerId==3):
            pg.display.update()
            img = pg.surfarray.array3d(self.screen).transpose([1, 0, 2])
        else:

            img = np.zeros((1,1,3))
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
