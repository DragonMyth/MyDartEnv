import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from gym.envs.flex import flex_env
import pygame as pg

try:
    import bindings as pyFlex
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: PyFlex Binding is not installed correctly)".format(e))


class PlasticSpringMultiGoalReshapingSolidEnv(flex_env.FlexEnv):
    def __init__(self):

        self.resolution = 32
        obs_size = self.resolution * self.resolution * 4 + 8

        self.frame_skip = 10
        action_bound = np.array([[-4, -4, -1, -1], [4, 4, 1, 1]])
        obs_high = np.ones(obs_size) * np.inf
        obs_low = -obs_high
        observation_bound = np.array([obs_low, obs_high])
        flex_env.FlexEnv.__init__(self, self.frame_skip, obs_size, observation_bound, action_bound, scene=4,disableViewer=True)
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.action_scale = (action_bound[1] - action_bound[0]) / 2
        self.barDim = np.array([1.5,1,0.01])

        self.center_list = np.array([[0, 2], [0, -2]])
        # self.center_list = np.array([[1.5, 1.5], [-1.5, -1.5]])

        # self.center_list = np.array([[0,0]])

        # self.center_list = np.random.uniform(-2, 2, (100, 2))

        self.randGoalRange = self.center_list.shape[0]

        self.circle_center = np.tile(np.random.choice(self.randGoalRange, size=2, replace=False),
                                     (self.numInstances, 1))

        # self.goal_gradients = np.zeros((self.numInstances,self.resolution,self.resolution))
        self.global_rot = self.generate_rand_rot_vec()
        self.partIdxGoal1 = None
        self.partIdxGoal2 = None

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

    def _step(self, action):
        action = action * self.action_scale
        centers = self.center_list[self.circle_center]

        group1_center = centers[:, 0]
        group2_center = centers[:, 1]

        prev_state = self.get_state()
        prev_group1_parts = prev_state[:, 4::][self.partIdxGoal1]
        prev_group1_parts = np.reshape(prev_group1_parts, (
            self.numInstances, int(prev_group1_parts.shape[0] / self.numInstances), prev_group1_parts.shape[1]))

        prev_group2_parts = prev_state[:, 4::][self.partIdxGoal2]
        prev_group2_parts = np.reshape(prev_group2_parts, (
            self.numInstances, int(prev_group2_parts.shape[0] / self.numInstances), prev_group2_parts.shape[1]))

        expanded_group1_centers = np.expand_dims(group1_center, axis=1)
        expanded_group1_centers = np.repeat(expanded_group1_centers, prev_group1_parts.shape[1], axis=1)

        expanded_group2_centers = np.expand_dims(group2_center, axis=1)
        expanded_group2_centers = np.repeat(expanded_group2_centers, prev_group2_parts.shape[1], axis=1)

        prev_distance_group1 = np.sum(np.linalg.norm(prev_group1_parts - expanded_group1_centers, axis=2), axis=1)
        prev_distance_group2 = np.sum(np.linalg.norm(prev_group2_parts - expanded_group2_centers, axis=2), axis=1)

        prev_var_group1 = np.var(prev_group1_parts, axis=(1, 2))
        prev_var_group2 = np.var(prev_group2_parts, axis=(1, 2))

        prev_mean_group1 = np.mean(prev_group1_parts, axis=(1))
        prev_mean_group2 = np.mean(prev_group2_parts, axis=(1))
        prev_mean_dist = np.clip(np.linalg.norm((prev_mean_group1-prev_mean_group2),axis=1),0,4)

        # prev_distance_group1 = np.linalg.norm(prev_mean_group1 - centers[:, 0], axis=1)
        # prev_distance_group2 = np.linalg.norm(prev_mean_group2 - centers[:, 1], axis=1)

        for i in range(action.shape[0]):
            targ_pos_trans = np.matmul(action[i, 0:2].transpose(), self.global_rot[i]).transpose()
            targ_rot_trans = np.matmul(action[i, 2:4].transpose(), self.global_rot[i]).transpose()

            action[i, 0:2] = targ_pos_trans
            action[i, 2:4] = targ_rot_trans

        solid = np.zeros((self.numInstances,1))
        action = np.concatenate([action, solid], axis=1)

        done = self.do_simulation(action, self.frame_skip)

        curr_state = self.get_state()

        curr_group1_parts = curr_state[:, 4::][self.partIdxGoal1]
        curr_group1_parts = np.reshape(curr_group1_parts, (
            self.numInstances, int(curr_group1_parts.shape[0] / self.numInstances), curr_group1_parts.shape[1]))

        curr_group2_parts = curr_state[:, 4::][self.partIdxGoal2]
        curr_group2_parts = np.reshape(curr_group2_parts, (
            self.numInstances, int(curr_group2_parts.shape[0] / self.numInstances), curr_group2_parts.shape[1]))

        curr_distance_group1 = np.sum(np.linalg.norm(curr_group1_parts - expanded_group1_centers, axis=2), axis=1)
        curr_distance_group2 = np.sum(np.linalg.norm(curr_group2_parts - expanded_group2_centers, axis=2), axis=1)

        curr_var_group1 = np.var(curr_group1_parts, axis=(1, 2))
        curr_var_group2 = np.var(curr_group2_parts, axis=(1, 2))

        curr_mean_group1 = np.mean(curr_group1_parts, axis=(1))
        curr_mean_group2 = np.mean(curr_group2_parts, axis=(1))

        curr_mean_dist = np.clip(np.linalg.norm((curr_mean_group1-curr_mean_group2),axis=1),0,4)
        # curr_distance_group1 = np.linalg.norm(curr_mean_group1 - centers[:, 0], axis=1)
        # curr_distance_group2 = np.linalg.norm(curr_mean_group2 - centers[:, 1], axis=1)
        obs = self._get_obs()

        group1_rwd_distannce = 0.01*(prev_distance_group1 - curr_distance_group1)
        group2_rwd_distannce = 0.01*(prev_distance_group2 - curr_distance_group2)

        # if(curr_mean_dist>=3):
        group1_rwd_var =0*((prev_var_group1 - curr_var_group1))
        group2_rwd_var = 0*((prev_var_group2 - curr_var_group2))

        separation_rwd = (curr_mean_dist-prev_mean_dist)

        rewards = group1_rwd_distannce + group2_rwd_distannce + group1_rwd_var + group2_rwd_var+separation_rwd

        info = {'Total Reward': np.mean(rewards), "Distance 1": np.mean(group1_rwd_distannce),
                "Var 1": np.mean(group1_rwd_var),"Distance 2": np.mean(group2_rwd_distannce),
                "Var 2": np.mean(group2_rwd_var)}
        return obs, rewards, done, info

    def _get_obs(self):

        states = self.get_state()
        obs_list = []

        for i in range(self.numInstances):
            state = states[i]
            part_state = state[4::]

            part_state_group1 = part_state[self.partIdxGoal1[i]]
            part_state_group2 = part_state[self.partIdxGoal2[i]]

            bar_state = state[:4]

            bar_pos_trans = np.matmul(bar_state[0].transpose(), self.global_rot[i].transpose()).transpose()
            bar_rot_trans = np.matmul(bar_state[1].transpose(), self.global_rot[i].transpose()).transpose()

            bar_vel_trans = np.matmul(bar_state[2].transpose(), self.global_rot[i].transpose()).transpose()

            bar_state[0] = bar_pos_trans
            bar_state[1] = bar_rot_trans
            bar_state[2] = bar_vel_trans

            bar_density = self.get_voxel_bar_density(bar_state, self.global_rot[i])
            density_group1 = self.get_particle_density(part_state_group1, self.global_rot[i], normalized=True)
            density_group2 = self.get_particle_density(part_state_group2, self.global_rot[i], normalized=True)

            goal_gradient = self.get_goal_gradient(self.center_list[self.circle_center[i]], self.global_rot[i])

            obs = np.concatenate(
                [bar_state.flatten(), density_group1.flatten(), density_group2.flatten(), goal_gradient.flatten(),
                 bar_density.flatten()])

            obs_list.append(obs)

        return np.array(obs_list)

    def get_goal_gradient(self, goal, global_rot):

        x, y = np.meshgrid(np.linspace(-4, 4, self.resolution), np.linspace(-4, 4, self.resolution))
        sigma = 0.3
        gradient = np.zeros(x.shape)
        for i in range(goal.shape[0]):
            goal_rot = np.matmul(goal[i].transpose(), global_rot.transpose()).transpose()
            gradient += np.exp(-(((x - goal_rot[0]) ** 2 + (y - goal_rot[1]) ** 2) / (2.0 * sigma ** 2)))

        return gradient

    def get_voxel_bar_density(self, bar_state, global_rot):

        center = bar_state[0]
        direction = bar_state[1].copy()
        direction[1] = -direction[1]
        ## half length is 1.5
        end_point_1 = center + direction * 1.5
        end_point_2 = center - direction * 1.5

        step = 1.0 / 40
        interp = np.arange(0, 1 + step, step)
        interp = np.expand_dims(interp, axis=1)
        interp = np.repeat(interp, 2, axis=1)
        interp = (1 - interp) * end_point_1 + interp * end_point_2
        interp_x = interp[:, 0]
        interp_y = interp[:, 1]

        H, xedges, yedges = np.histogram2d(interp_y, interp_x, bins=[self.resolution, self.resolution],
                                           range=[[-4, 4], [-4, 4]])
        H = (H > 0).astype(int)

        return H

    def get_particle_density(self, particles, global_rot, normalized=True):

        particles_rot = np.matmul(particles, global_rot.transpose())
        x_pos = particles_rot[:, 0]
        y_pos = particles_rot[:, 1]

        H, xedges, yedges = np.histogram2d(y_pos, x_pos, bins=[self.resolution, self.resolution],
                                           range=[[-4, 4], [-4, 4]])

        if normalized:
            H = H ** (1.0 / 2)
            H = H / 15

        return H

    def get_state(self):
        full_state = flex_env.FlexEnv.get_state(self)
        return full_state[:, :, (0, 2)]

    def _reset(self):
        flex_env.FlexEnv._reset(self)

        self.global_rot = self.generate_rand_rot_vec()

        self.circle_center = np.tile(np.random.choice(self.randGoalRange, size=2, replace=False),
                                     (self.numInstances, 1))

        goals= self.center_list.flatten()
        self.set_goal(np.tile(goals,(self.numInstances,1)))


        pos = np.random.uniform(-3,3,(self.numInstances,2))
        rot = np.random.uniform(-np.pi,np.pi,(self.numInstances,1))

        # pos = np.random.uniform(-0.0,0.0,(self.numInstances,2))
        # rot = np.random.uniform(-0.1*np.pi,0.1*np.pi,(self.numInstances,1))

        vel = np.zeros((self.numInstances,2))
        angVel = np.zeros((self.numInstances,1))
        barDim = np.tile(self.barDim,(self.numInstances,1))

        controllers = np.concatenate([pos,rot,vel,angVel,barDim],axis=1)
        self.set_controller(controllers)

        states = self.get_state()
        self.partIdxGoal1 = states[:, 4::, 1] > 0
        self.partIdxGoal2 = states[:, 4::, 1] <= 0

        return self._get_obs()

    def _render(self, mode='human', close=False):
        if (self.disableViewer or close):
            return
        else:
            if not self.screen:
                pg.init()
                self.screen = pg.display.set_mode(self.screen_size, display=pg.OPENGL)
            width = self.screen_size[0]
            height = self.screen_size[1]
            gap = self.sub_screen_gap

            tl_surface = pg.Surface((width / 2 - gap / 2, height / 2 - gap / 2))
            tr_surface = pg.Surface((width / 2 - gap / 2, height / 2 - gap / 2))
            ll_surface = pg.Surface((width / 2 - gap / 2, height / 2 - gap / 2))
            lr_surface = pg.Surface((width / 2 - gap / 2, height / 2 - gap / 2))
            self.pygame_draw([tl_surface, tr_surface, ll_surface, lr_surface])
            return flex_env.FlexEnv._render(self)

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

        part_map_1 = obs[0, 8:8 + self.resolution * self.resolution]
        part_map_2 = obs[0, 8 + self.resolution * self.resolution:8 + 2 * self.resolution * self.resolution]
        goal_map = obs[0, 8 + 2 * self.resolution * self.resolution:8 + 3 * (self.resolution * self.resolution)]
        bar_map = obs[0, 8 + 3 * self.resolution * self.resolution:8 + 4 * (self.resolution * self.resolution)]

        bar_map = np.reshape(bar_map, (self.resolution, self.resolution)).astype(np.float64)
        goal_map = np.reshape(goal_map, (self.resolution, self.resolution)).astype(np.float64)
        part_map_1 = np.reshape(part_map_1, (self.resolution, self.resolution)).astype(np.float64)
        part_map_2 = np.reshape(part_map_2, (self.resolution, self.resolution)).astype(np.float64)

        self.draw_grid(tl, bar_map, 0, 1)
        self.draw_grid(tr, goal_map, 0, 1)

        self.draw_grid(lr, part_map_1, 0, 1)
        self.draw_grid(ll, part_map_2, 0, 1)

        self.screen.blit(tl, (0, 0))
        self.screen.blit(tr, (self.screen.get_width() / 2 + self.sub_screen_gap / 2, 0))
        self.screen.blit(ll, (0, self.screen.get_height() / 2 + self.sub_screen_gap / 2))

        self.screen.blit(lr, (
            self.screen.get_width() / 2 + self.sub_screen_gap / 2,
            self.screen.get_height() / 2 + self.sub_screen_gap / 2))

    def draw_grid(self, surface, data, min, scale):
        data = (data - min) / scale
        w_gap = surface.get_width() / data.shape[0]
        h_gap = surface.get_height() / data.shape[1]

        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                color = np.array([1.0, 1.0, 1.0])
                color *= data[y, x]
                color = np.clip(color, 0, 1)

                final_color = 255 * (np.array([1, 0, 0]) * color + np.array([0, 0, 1]) * (1 - color))
                pg.draw.rect(surface, final_color,
                             pg.Rect(x * w_gap, y * h_gap, (x + 1) * w_gap, (y + 1) * h_gap))


if __name__ == '__main__':
    env = PlasticSpringMultiGoalReshapingSolidEnv()

    env.reset()
    for i in range(2000):
        # env.render()
        # print(pyFlex.get_state())
        # act = np.random.uniform([-4, -4, -1, -1], [4, 4, 1, 1],(25,4))
        act = np.zeros((16, 4))
        act[:, -1] = 0
        obs, rwd, done, info = env.step(act)

        if i % 100 == 0:
            print(i)
        if done:
            break
    # else:
    #     continue
    # break
