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


class PlasticSpringReshapingEnv(flex_env.FlexEnv):
    def __init__(self):

        self.resolution = 32
        obs_size = self.resolution * self.resolution * 3 + 8

        self.frame_skip = 10
        action_bound = np.array([[-4, -4, -1, -1, -1], [4, 4, 1, 1, 1]])
        obs_high = np.ones(obs_size) * np.inf
        obs_low = -obs_high
        observation_bound = np.array([obs_low, obs_high])
        flex_env.FlexEnv.__init__(self, self.frame_skip, obs_size, observation_bound, action_bound, scene=4,disableViewer=False)
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.action_scale = (action_bound[1] - action_bound[0]) / 2

        # self.center_list = np.array([[1.5,1.5],[-1.5,-1.5],[-1.5,1.5],[1.5,-1.5]])

        # self.center_list = np.array([[0,1.5],[0,-1.5]])
        self.center_list = np.array([[0,0]])

        # self.center_list = np.random.uniform(-2, 2, (100, 2))

        self.randGoalRange = self.center_list.shape[0] - 1

        self.circle_center = np.random.random_integers(0, self.randGoalRange, self.numInstances)

        # self.goal_gradients = np.zeros((self.numInstances,self.resolution,self.resolution))
        self.global_rot = self.generate_rand_rot_vec()

    def generate_rand_rot_vec(self):
        rand_rot_ang = np.random.uniform(-np.pi, np.pi, self.numInstances)
        # rand_rot_ang = np.ones(self.numInstances)
        rand_rot_ang=0

        rand_rot_vec = np.ones((self.numInstances, 2, 2))

        rand_rot_vec[:, 0, 0] = np.cos(rand_rot_ang)
        rand_rot_vec[:, 0, 1] = -np.sin(rand_rot_ang)
        rand_rot_vec[:, 1, 0] = np.sin(rand_rot_ang)
        rand_rot_vec[:, 1, 1] = np.cos(rand_rot_ang)
        return rand_rot_vec

    def _step(self, action):
        action = action * self.action_scale
        prev_state = self.get_state()
        centers = self.center_list[self.circle_center]

        expanded_centers = np.expand_dims(centers, axis=1)
        expanded_centers = np.repeat(expanded_centers, prev_state.shape[1], axis=1)

        # prev_distance = 0.1 * np.sum(np.linalg.norm(prev_state - expanded_centers, axis=2)[:, 4::] ** 3, axis=1)
        prev_var = np.var(prev_state[:, 4::], axis=(1, 2))

        for i in range(action.shape[0]):
            targ_pos_trans = np.matmul(action[i, 0:2].transpose(), self.global_rot[i]).transpose()
            targ_rot_trans = np.matmul(action[i, 2:4].transpose(), self.global_rot[i]).transpose()

            action[i, 0:2] = targ_pos_trans
            action[i, 2:4] = targ_rot_trans

        action = np.concatenate([action, centers,centers], axis=1)

        done = self.do_simulation(action, self.frame_skip)

        curr_state = self.get_state()
        curr_var = np.var(curr_state[:, 4::], axis=(1, 2))

        # curr_distance = 0.1 * np.sum(np.linalg.norm(curr_state - expanded_centers, axis=2)[:, 4::] ** 3, axis=1)

        obs = self._get_obs()

        # rewards = (prev_distance - curr_distance)

        rewards = (prev_var-curr_var)
        info = {'Total Reward': rewards[0], }

        return obs, rewards, done, info


    def _get_obs(self):

        states = self.get_state()
        obs_list = []

        for i in range(self.numInstances):
            state = states[i]
            part_state = state[4::]

            bar_state = state[:4]

            bar_pos_trans = np.matmul(bar_state[0].transpose(), self.global_rot[i].transpose()).transpose()
            bar_rot_trans = np.matmul(bar_state[1].transpose(), self.global_rot[i].transpose()).transpose()

            bar_vel_trans = np.matmul(bar_state[2].transpose(), self.global_rot[i].transpose()).transpose()

            bar_state[0] = bar_pos_trans
            bar_state[1] = bar_rot_trans
            bar_state[2] = bar_vel_trans

            bar_density = self.get_voxel_bar_density(bar_state, self.global_rot[i])
            density = self.get_particle_density(part_state, self.global_rot[i], normalized=True)
            goal_gradient = self.get_goal_gradient(self.center_list[self.circle_center[i]], self.global_rot[i])

            obs = np.concatenate(
                [bar_state.flatten(), density.flatten(), goal_gradient.flatten(), bar_density.flatten()])

            obs_list.append(obs)

        return np.array(obs_list)

    def get_goal_gradient(self, goal, global_rot):

        goal_rot = np.matmul(goal.transpose(), global_rot.transpose()).transpose()

        x, y = np.meshgrid(np.linspace(-4, 4, self.resolution), np.linspace(-4, 4, self.resolution))
        sigma = 0.5

        gradient = np.exp(-(((x - goal_rot[0]) ** 2 + (y - goal_rot[1]) ** 2) / (2.0 * sigma ** 2)))
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
            H=H/10

        return H

    def get_state(self):
        full_state = flex_env.FlexEnv.get_state(self)
        return full_state[:, :, (0, 2)]


    def _reset(self):
        flex_env.FlexEnv._reset(self)

        self.global_rot = self.generate_rand_rot_vec()
        self.circle_center = np.random.random_integers(0, self.randGoalRange, self.numInstances)

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

            tl_surface = pg.Surface((width / 2-gap/2, height / 2-gap/2))
            tr_surface = pg.Surface((width / 2-gap/2, height / 2-gap/2))
            ll_surface = pg.Surface((width / 2-gap/2, height / 2-gap/2))
            lr_surface = pg.Surface((width / 2-gap/2, height / 2-gap/2))
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

        bar_map = obs[0, -self.resolution * self.resolution::]
        goal_map = obs[0, 8 + self.resolution * self.resolution:8+2*(self.resolution * self.resolution)]

        part_map = obs[0, 8:8 + self.resolution * self.resolution]

        bar_map = np.reshape(bar_map, (self.resolution, self.resolution)).astype(np.float64)
        goal_map = np.reshape(goal_map, (self.resolution, self.resolution)).astype(np.float64)
        part_map = np.reshape(part_map, (self.resolution, self.resolution)).astype(np.float64)

        self.draw_grid(tl, bar_map, 0, 1)
        self.draw_grid(tr, goal_map, 0, 1)

        self.draw_grid(lr, part_map, 0, 1)

        self.screen.blit(tl, (0, 0))
        self.screen.blit(tr, (self.screen.get_width() / 2 + self.sub_screen_gap / 2, 0))
        self.screen.blit(ll, (0, self.screen.get_height() / 2 + self.sub_screen_gap / 2))

        self.screen.blit(lr, (
        self.screen.get_width() / 2 + self.sub_screen_gap / 2, self.screen.get_height() / 2 + self.sub_screen_gap / 2))

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
    env = PlasticSpringReshapingEnv()

    env.reset()
    for _ in range(1000):
        # print(pyFlex.get_state())
        # act = np.random.uniform([-4, -4, -1, -1], [4, 4, 1, 1],(25,4))
        act = np.zeros((1, 5))
        act[:, -1] = 0
        obs, rwd, done, info = env.step(act)
        if done:
            break
    # else:
    #     continue
    # break
