import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from gym.envs.flex import flex_env
import pygame as pg
import itertools
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
try:
    import bindings as pyFlex
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: PyFlex Binding is not installed correctly)".format(e))


class PlasticSpreadingRotHeightParticleCentricActEnv(flex_env.FlexEnv):
    def __init__(self):

        self.resolution = 32
        self.direct_info_dim = 14
        obs_size = self.resolution * self.resolution * 2 + self.direct_info_dim

        self.frame_skip = 10
        self.mapHalfExtent = 4
        self.mapPartitionSize = 3
        self.idxPool = np.array([x for x in itertools.product(np.arange(self.mapPartitionSize) - int(
            self.mapPartitionSize / 2), np.arange(self.mapPartitionSize) - int(self.mapPartitionSize / 2))])

        self.numInitClusters = 1
        self.randomCluster = True
        self.clusterDim = np.array([6,6,6])
        action_bound = np.array([[-7, -20,-7, -np.pi / 2], [
            7,20, 7 ,np.pi / 2]])

        obs_high = np.ones(obs_size) * np.inf
        obs_low = -obs_high
        observation_bound = np.array([obs_low, obs_high])
        flex_env.FlexEnv.__init__(self, self.frame_skip, obs_size, observation_bound, action_bound, scene=2, viewer=0)

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.action_scale = (action_bound[1] - action_bound[0]) / 2
        self.barDim = np.array([0.7, 0.5, 0.01])

        # self.goal_gradients = np.zeros((self.numInstances,self.resolution,self.resolution))
        self.global_rot = self.generate_rand_rot_vec()

        self.initClusterparam = np.zeros(
            (self.numInstances, 6 * self.numInitClusters))

        self.rolloutCnt = 0
        self.stage = np.ones(self.numInstances)
        self.rolloutRet = np.zeros(self.numInstances)
        self.currCurriculum =0
        self.rwdBuffer=[[0,0,0] for _ in range(100)]
        self.curr_pc = np.array([[1.0,1.0],[-1.0,-1.0]])
        print("With Height Map Attraction Target Centric Action")
    def generate_rand_rot_vec(self):
        rand_rot_ang = np.random.uniform(-np.pi, np.pi, self.numInstances)
        # rand_rot_ang = np.ones(self.numInstances)
        rand_rot_ang = 0

        rand_rot_vec = np.ones((self.numInstances, 2, 2))

        rand_rot_vec[:, 0, 0] = np.cos(rand_rot_ang)
        rand_rot_vec[:, 0, 1] = -np.sin(rand_rot_ang)
        rand_rot_vec[:, 1, 0] = np.sin(rand_rot_ang)
        rand_rot_vec[:, 1, 1] = np.cos(rand_rot_ang)
        return rand_rot_vec

    def angle_to_rot_matrix(self, angles):
        rot_vec = np.ones((self.numInstances, 2, 2))

        rot_vec[:, 0, 0] = np.cos(angles)
        rot_vec[:, 0, 1] = -np.sin(angles)
        rot_vec[:, 1, 0] = np.sin(angles)
        rot_vec[:, 1, 1] = np.cos(angles)
        return rot_vec

    def _step(self, action):
        action = action * self.action_scale
        prev_bar_state,prev_part_state,prev_part_heights = self.get_state()

        rot_mat = self.angle_to_rot_matrix(action[:, 3])

        transformed_action = np.zeros((self.numInstances, 6))

        heighest_idx = np.argmax(prev_part_heights,axis=1)


        for i in range(action.shape[0]):
            bar_rot_trans = prev_bar_state[i, 1,(0,2)]
            bar_rot_vec = bar_rot_trans / np.linalg.norm(bar_rot_trans)
            bar_rot = np.zeros((2, 2))
            bar_rot[0, 0] = bar_rot_vec[0]
            bar_rot[0, 1] = -bar_rot_vec[1]
            bar_rot[1, 0] = bar_rot_vec[1]
            bar_rot[1, 1] = bar_rot_vec[0]

            targ_pos_trans = np.matmul(
                bar_rot.transpose(),action[i, (0,2)])

            action[i, (0,2)] = targ_pos_trans

            heighest_xz_pos = prev_part_state[i,heighest_idx[i]]
            heighest_y_pos = prev_part_heights[i,heighest_idx[i]]

            transformed_action[i, 0:3] = action[i, 0:3] + np.array([heighest_xz_pos[0],heighest_y_pos,heighest_xz_pos[1]])
            transformed_action[i, 3:5] = np.matmul(prev_bar_state[i, 1,(0,2)], rot_mat[i].transpose())

        flex_action = np.zeros((self.numInstances,6))
        flex_action[:,0] = transformed_action[:,0]

        flex_action[:,1] = transformed_action[:,1]
        

        flex_action[:,2] = transformed_action[:,2]
        flex_action[:,3] = transformed_action[:,3]
        flex_action[:,4] = transformed_action[:,4]
        flex_action[:,5] = -1
       
        prev_obs = self._get_obs()
        

        prev_height_sum = (np.sum(prev_part_heights,axis=1))

        #Simulation 
        done = self.do_simulation(flex_action, self.frame_skip)

        curr_bar_state,curr_part_state,curr_part_heights = self.get_state()

        obs = self._get_obs()

        
        part_movement_rwd =  np.mean(np.linalg.norm(
            (curr_part_state - prev_part_state), axis=2), axis=1) 
        part_movement_rwd=(1-np.exp(-40*part_movement_rwd))
        curr_height_sum = (np.max(curr_part_heights,axis=1))

        target_dist_curr = np.zeros(self.numInstances)
        to_bar_dist_curr = np.zeros(self.numInstances)
        # for i in range(self.numInstances):
        #     dist= to_bar_dist_curr[i]
        #     if(dist<=2):
        #         self.stage[i]  =  0
        #         target_dist_curr[i] = 0.6*np.clip(np.exp(0.01*(-curr_height_sum[i])),0,1)+0.4*(1-np.exp(-40*part_movement_rwd[i]))

        #     else:
        #         self.stage[i]  =  1
        #         target_dist_curr[i] = -0.1*np.exp(0.02*(dist-2))

        height_min_rwd = np.clip(np.exp(3*(-curr_height_sum)),0,1)
        rewards =0.8*height_min_rwd+0.2*part_movement_rwd

        # print(self.stage[0])
        self.rolloutRet+=rewards
        info = {
            'Total Reward': rewards[0],

        }
        reward_decomp = [rewards[0],0.8*height_min_rwd[0],0.2*part_movement_rwd[0]]

        # print(self.stage[0])
        self.rolloutRet+=rewards
        info = {
            'Total Reward': rewards[0],

        }

        if(len(self.rwdBuffer)>=100):
            self.rwdBuffer.pop(0)
        self.rwdBuffer.append(reward_decomp)
        return obs, rewards, done, info

    def _get_obs(self):

        bar_states,part_states,part_heights = self.get_state()
        obs_list = []

        for i in range(self.numInstances):
            stage = self.stage[i]
            part_state = part_states[i]
            part_state_adjust = part_state[(part_state[:,0]>-self.mapHalfExtent) & (part_state[:,0]<self.mapHalfExtent)&(part_state[:,1]>-self.mapHalfExtent)&(part_state[:,1]<self.mapHalfExtent)]
            part_height = part_heights[i]
            part_height_adjust = part_height[(part_state[:,0]>-self.mapHalfExtent) & (part_state[:,0]<self.mapHalfExtent)&(part_state[:,1]>-self.mapHalfExtent)&(part_state[:,1]<self.mapHalfExtent)]
            bar_state = bar_states[i]

            bar_rot_vec = bar_state[1,(0,2)] / np.linalg.norm(bar_state[1,(0,2)])

            bar_rot = np.zeros((2, 2))
            bar_rot[0, 0] = bar_rot_vec[0]
            bar_rot[0, 1] = -bar_rot_vec[1]
            bar_rot[1, 0] = bar_rot_vec[1]
            bar_rot[1, 1] = bar_rot_vec[0]


            heighest_idx = np.argmax(part_height_adjust)

            heighest_xz_pos = part_state_adjust[heighest_idx]
            heighest_y_pos = part_height_adjust[heighest_idx]

            density = self.get_particle_density(
                part_state_adjust, bar_state, bar_rot, normalized=True)

            height_map = self.get_mean_height_map(part_state_adjust,bar_state,bar_rot,part_height_adjust)

            obs = np.concatenate(
                [bar_state[(0,2),:].flatten(),bar_state[1,(0,2)].flatten(),bar_state[3,(0,2)].flatten() ,[heighest_xz_pos[0],heighest_y_pos,heighest_xz_pos[1]],[stage], density.flatten(),height_map.flatten()
                 ])
            obs_list.append(obs)

        return np.array(obs_list)

    def get_goal_gradient(self, goal, bar_state, global_rot):

        x, y = np.meshgrid(np.linspace(-self.mapHalfExtent, self.mapHalfExtent, self.resolution),
                           np.linspace(-self.mapHalfExtent, self.mapHalfExtent, self.resolution))
        sigma = 0.3
        # sigma = 4

        gradient = np.zeros(x.shape)
        for i in range(goal.shape[0]):
            # print(goal[i])
            # print(bar_state[0])
            goal[i] -= bar_state[0]

            goal_rot = np.matmul(goal[i].transpose(),
                                 global_rot.transpose()).transpose()
            goal_rot = np.clip(goal_rot, -self.mapHalfExtent, self.mapHalfExtent)
            gradient += np.exp(-(((x - goal_rot[0]) ** 2 +
                                  (y - goal_rot[1]) ** 2) / (2.0 * sigma ** 2)))

        return gradient

    def get_voxel_bar_density(self, bar_state, global_rot):

        center = np.zeros(2)
        direction = bar_state[1].copy()
        direction[1] = -direction[1]
        # half length is 1.5
        end_point_1 = center + direction * self.barDim[0]
        end_point_2 = center - direction * self.barDim[0]

        step = 1.0 / 100
        interp = np.arange(0, 1 + step, step)
        interp = np.expand_dims(interp, axis=1)
        interp = np.repeat(interp, 2, axis=1)
        interp = (1 - interp) * end_point_1 + interp * end_point_2
        interp_x = interp[:, 0]
        interp_y = interp[:, 1]

        H = self.get_density(interp, self.resolution,
                             1.5, self.mapHalfExtent) / 100
        H = np.clip(H, 0, 1)
        return H

    def get_particle_density(self, particles, bar_state, global_rot, normalized=True,width=2.5):
        if(particles.shape[0] == 0):
            return np.zeros((self.resolution,self.resolution))
        particles -= bar_state[0,(0,2)]

        particles = np.matmul(particles, global_rot.transpose())

        particles = np.clip(particles, -self.mapHalfExtent, self.mapHalfExtent)
        # H = self.get_density(particles, self.resolution,
        #                      2.5, self.mapHalfExtent)
        H = self.get_density(particles, self.resolution,
                        width, self.mapHalfExtent)
        x_pos = particles[:, 0]
        y_pos = particles[:, 1]
        if normalized:
            # H = H ** (1.0 / 2)
            H = H / (200)
            # H = H / (50)

            H = np.clip(H, 0, 1)
        return H
    def get_mean_height_map(self,particles,bar_state,rot,heights, normalized=True,width=2.5):
        if(particles.shape[0] == 0):
            return np.zeros((self.resolution,self.resolution))
        particles -= bar_state[0,(0,2)]

        particles = np.matmul(particles, rot.transpose())

        particles = np.clip(particles, -self.mapHalfExtent, self.mapHalfExtent)
        # H = self.get_density(particles, self.resolution,
        #                      2.5, self.mapHalfExtent)
        H = self.get_height_map(particles,heights,self.resolution,width,self.mapHalfExtent)
        # print(np.max(H))
        # if normalized:
            # H = H ** (1.0 / 2)
            # H = H / (10)
            # H = H / (50)

            # H = np.clip(H, 0, 1)
        return H
    def get_state(self):
        full_state = flex_env.FlexEnv.get_state(self)
        part_state = full_state[:,4::,(0,2)]
        bar_state = full_state[:,:4,:]
        part_heights = full_state[:,4::,1]
        return bar_state,part_state,part_heights

    def _reset(self):
        self.rwdBuffer=[[0,0,0] for _ in range(100)]

        if(np.mean(self.rolloutRet) > 400):
            self.currCurriculum=min(3,self.currCurriculum+1)
            
        
        print("Current Curriculum Level: ", self.currCurriculum)     
        print("Current Cluster Number Level: ", self.numInitClusters)         
        print("Return at current rollout: ", self.rolloutRet)            
        print("Mean Return at current rollout: ", np.mean(self.rolloutRet))            

        # self.randGoalRange = 2+2*min(3,self.currCurriculum)
        # self.numInitClusters = 1+self.currCurriculum
        self.rolloutRet = np.zeros(self.numInstances)
        if self.randomCluster:
            # self.idxPool = np.array([[-1, -1],[1, 1]])
            # self.idxPool = np.array([[-1, -1]])

            self.idxPool = np.array([[0,0]])

            # self.idxPool = np.array([x for x in itertools.product(np.arange(self.mapPartitionSize) - int(
            #     self.mapPartitionSize / 2), np.arange(self.mapPartitionSize) - int(self.mapPartitionSize / 2))])

            # Pre-flex reset calculation
            self.initClusterparam = np.zeros(
                (self.numInstances, 6 * self.numInitClusters))

            for i in range(self.numInstances):

                indices = np.random.choice(
                    np.arange(self.idxPool.shape[0]), size=self.numInitClusters, replace=False)

                for j in range(self.numInitClusters):
                    self.initClusterparam[i, (j * 6, j * 6 + 2)
                    ] = self.idxPool[indices[j]] * 2.5
                    self.initClusterparam[i, j * 6 + 3:j * 6 + 6] = self.clusterDim

                self.setInitClusterParam(self.initClusterparam)

        flex_env.FlexEnv._reset(self)

        # Post-flex reset calculation
        self.global_rot = self.generate_rand_rot_vec()

    


        self.setMapHalfExtent(self.mapHalfExtent)
        #
        pos = np.random.uniform(-self.mapHalfExtent, self.mapHalfExtent, (self.numInstances, 2))
        pos_y = np.random.uniform(0,1,(self.numInstances,1))
        rot = np.random.uniform(-np.pi, np.pi, (self.numInstances, 1))

        # pos = np.zeros((self.numInstances, 2))
        # pos[:, 0] = -1.8
        # pos[:, 1] = -1.8
        # rot = np.zeros((self.numInstances,1))
        #rot[:,0] = np.pi/4


        vel = np.random.uniform(-1, 1, (self.numInstances, 3))
        angVel = np.random.uniform(-0.1, 0.1, (self.numInstances, 1))
        barDim = np.tile(self.barDim, (self.numInstances, 1))

        controllers = np.concatenate([pos[:,0][:,np.newaxis],pos_y,pos[:,1][:,np.newaxis], rot, vel, angVel, barDim], axis=1)
        self.set_controller(controllers)
        self.rolloutCnt += 1
        return self._get_obs()

    def _render(self, mode='human', close=False):
        if(self.viewerId==2 or self.viewerId==3): 
            if(self.viewerId==2 and not self.screen):
                pg.init()
                self.screen = pg.display.set_mode(self.screen_size, DOUBLEBUF | OPENGL)
            elif(self.viewerId==3 and not self.screen):
                pg.init()
                self.screen = pg.display.set_mode(self.screen_size)
            
            width = self.screen_size[0]
            height = self.screen_size[1]
            gap = self.sub_screen_gap
            if(self.viewerId==3):
                tl_surface = pg.Surface(
                    (width / 2 - gap / 2, height / 2 - gap / 2))
                tr_surface = pg.Surface(
                    (width / 2 - gap / 2, height / 2 - gap / 2))
                ll_surface = pg.Surface(
                    (width / 2 - gap / 2, height / 2 - gap / 2))
                lr_surface = pg.Surface(
                    (width / 2 - gap / 2, height / 2 - gap / 2))
                self.pygame_draw([tl_surface, tr_surface, ll_surface, lr_surface])
            else:
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                tl_surface = (0, 0, width / 2 - gap / 2, height / 2 - gap / 2)
                tr_surface = (width / 2 + gap / 2, 0, width / 2 - gap / 2, height / 2 - gap / 2)
                ll_surface = (0,height/2+gap/2,width / 2 - gap / 2, height / 2 - gap / 2)
                lr_surface = (width / 2 + gap / 2,height / 2 + gap / 2,width / 2 - gap / 2, height / 2 - gap / 2)
                self.pygame_draw_GL([tl_surface, tr_surface, ll_surface, lr_surface])

            return flex_env.FlexEnv._render(self)
    def pygame_draw_GL(self, surfaces):
        obs = self._get_obs()
        tl = surfaces[0]
        tr = surfaces[1]
        ll = surfaces[2]
        lr = surfaces[3]

        part_map = obs[0, 9:9 + self.resolution * self.resolution]
        goal_map = obs[0, 9 + self.resolution * self.resolution:9 +
                                                                2 * (self.resolution * self.resolution)]
        particle_goal_map = obs[0, 9 + 2*self.resolution * self.resolution:9 +3 * (self.resolution * self.resolution)]
        # bar_map = obs[0, 8 + 2 * self.resolution *
        #               self.resolution:8 + 3 * (self.resolution * self.resolution)]

        # bar_map = np.reshape(
        #     bar_map, (self.resolution, self.resolution)).astype(np.float64)
        goal_map = np.reshape(
            goal_map, (self.resolution, self.resolution)).astype(np.float64)
        part_map = np.reshape(
            part_map, (self.resolution, self.resolution)).astype(np.float64)
        particle_goal_map = np.reshape(
            particle_goal_map, (self.resolution, self.resolution)).astype(np.float64)
        # glBegin(GL_LINES)
        # glVertex2d(50,50)
        # glVertex2d(50, 100)
        # glVertex2d(100, 100)
        # glVertex2d(100, 50)
        # glEnd()
        # self.draw_grid_GL(tl, part_map, 0, 1)
        self.draw_grid_GL(tr, goal_map, 0, 1)

        self.draw_grid_GL(lr, part_map, 0, 1)

        self.draw_grid_GL(ll, particle_goal_map, 0, 1)
        #

    def pygame_draw(self, surfaces):
        obs = self._get_obs()
        tl = surfaces[0]
        tr = surfaces[1]
        ll = surfaces[2]
        lr = surfaces[3]
    
        tl.fill([200, 200, 200])
        tr.fill([200, 200, 200])
        ll.fill([200, 200, 200])
        lr.fill([200, 200, 200])
    
        part_map = obs[0, self.direct_info_dim:self.direct_info_dim + self.resolution * self.resolution]
       
        part_map = np.reshape(
            part_map, (self.resolution, self.resolution)).astype(np.float64)

        height_map =  obs[0, self.direct_info_dim + self.resolution * self.resolution:self.direct_info_dim + 2*self.resolution * self.resolution]
        height_map = np.reshape(
            height_map, (self.resolution, self.resolution)).astype(np.float64)
        # self.draw_grid(tl, bar_map, 0, 1)
        self.live_rwd(tl,self.rwdBuffer)
        # self.draw_grid(tr, goal_map, 0, 1)
        # self.live_pc(ll,self.curr_pc)
        self.draw_grid(ll, height_map, 0, 1)

        self.draw_grid(lr, part_map, 0, 1)
        #
        self.screen.blit(tl, (0, 0))
        self.screen.blit(tr, (self.screen.get_width() /
                              2 + self.sub_screen_gap / 2, 0))
        self.screen.blit(ll, (0, self.screen.get_height() /
                              2 + self.sub_screen_gap / 2))
    
        self.screen.blit(lr, (
            self.screen.get_width() / 2 + self.sub_screen_gap / 2,
            self.screen.get_height() / 2 + self.sub_screen_gap / 2))
    
    
    def live_rwd(self,surface,rwds):
        width = surface.get_width()
        height = surface.get_height()
        rwds = np.array(rwds)
        for j in range(rwds.shape[1]):
            rwd = rwds[:,j]
            # print(rwd)
            for i in range(len(rwd)-1):
                x0 = i/float(len(rwd))*width
                y0 = height/2-(rwd[i])*(height/2)

                x1 = (i+1)/float(len(rwd))*width
                y1 = height/2-(rwd[i+1])*(height/2)

                if j ==0:
                    color = 255* np.array([1,0,0])
                elif j==1:
                    color = 255* np.array([0,1,0])
                else:
                    color = 255* np.array([0,0,1])
                pg.draw.line(surface,color,(x0,y0),(x1,y1),1)
    def live_pc(self,surface,pc):
        width = surface.get_width()
        height = surface.get_height()
        color = 255*np.array([1,0,0])
        x0 = (pc[0,0]+self.mapHalfExtent)/(2*self.mapHalfExtent)*width
        y0 = (pc[0,1]+self.mapHalfExtent)/(2*self.mapHalfExtent)*height

        x1 = (pc[1,0]+self.mapHalfExtent)/(2*self.mapHalfExtent)*width
        y1 = (pc[1,1]+self.mapHalfExtent)/(2*self.mapHalfExtent)*height

        # print(x0,y0,x1,y1)
        pg.draw.line(surface,color,(x0,y0),(x1,y1),3)
    def draw_grid(self, surface, data, min, scale):
        data = (data - min) / scale
        w_gap = surface.get_width() / data.shape[0]
        h_gap = surface.get_height() / data.shape[1]

        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                color = np.array([1.0, 1.0, 1.0])
                color *= data[y, x]
                color = np.clip(color, 0, 1)
    
                final_color = 255 * \
                    (np.array([1, 0, 0]) * color +
                     np.array([0, 0, 1]) * (1 - color))
                pg.draw.rect(surface, final_color,
                             pg.Rect(x * w_gap, y * h_gap, (x + 1) * w_gap, (y + 1) * h_gap))

    def draw_grid_GL(self, surface, data, min, scale):
        data = (data - min) / scale
        w_gap = surface[2] / data.shape[0]
        h_gap = surface[3] / data.shape[1]
        offsetX = surface[0]
        offsetY = surface[1]

        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                color = np.array([1.0, 1.0, 1.0])
                color *= data[y, x]
                color = np.clip(color, 0, 1)

                final_color = (np.array([1, 0, 0]) * color +
                               np.array([0, 0, 1]) * (1 - color))

                glColor3d(final_color[0], final_color[1], final_color[2])
                glRectd(offsetX + x * w_gap, offsetY + y * h_gap, offsetX + (x + 1) * w_gap, offsetY + (y + 1) * h_gap)


if __name__ == '__main__':
    env = PlasticSpreadingRotHeightParticleCentricActEnv()

    env.reset()
    for i in range(2000):
        # env.render()
        # print(pyFlex.get_state())
        # act = np.random.uniform([-4, -4, -1, -1], [4, 4, 1, 1],(25,4))
        act = np.zeros((49, 4))
        # act[:, 0]=-1
        # act[:, 1] = 1
        # act[:, 2] = 1
        # act[:, 3] = 1

        # act[:, -1] = 1
        obs, rwd, done, info = env.step(act)
        env.render()
        if i % 100 == 0:
            print(i)
        if i % 100 == 0:
            env.reset()
        if done:
            # env.reset()
            break
