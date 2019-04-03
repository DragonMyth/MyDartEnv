"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import joblib
# from .Particle import Particle
from .ParticlesSim import ParticlesSim

logger = logging.getLogger(__name__)


class SimplerParticleCarvingRotation(gym.Env):
    def __init__(self):

        # 0 for path
        # 1 for wall
        # 2,3,4,5 for goal
        # 6,7,8,9 for hints
        self.dt = 0.02
        self.world_size = 2
        self.world_size_view = 2.5
        self.numCells = 32
        self.cellSize = self.world_size * 1.0 / self.numCells
        self.numParts = 500
        self.particleSim = ParticlesSim(self.numParts, self.dt)

        self.density_map = np.zeros((self.numCells, self.numCells), dtype=int)
        self.grid_world = [[[] for _ in range(len(self.density_map[0]))] for _ in range(len(self.density_map))]
        self.template = self.generate_template()
        self.geomCnt = 0
        self.frameskip = 5
        # self.obs_dim = self.numCells ** 2
        self.obs_dim = self.numCells * self.numCells + 8

        self.act_dim = 4
        # self.repelling_force_scale = 1
        self.action_high = np.ones(self.act_dim) * self.world_size_view / 2
        # self.action_high[2:4] = 1
        self.action_low = - self.action_high
        self.action_space = spaces.Box(self.action_low, self.action_high)

        self.pos_high = np.ones(3) * self.world_size_view / 2
        self.pos_high[2] = 2 * np.pi
        self.pos_low = - self.pos_high
        self.pos_low[2] = 0
        obs_high = np.ones(self.obs_dim) * self.world_size_view / 2
        obs_low = -obs_high
        self.observation_space = spaces.Box(obs_low, obs_high)
        self.t = 0

        self.curr_act = None
        self.action_scale = 1
        self.curr_iter = 1000
        self.half_length = 0.3
        self.curr_pos = np.array([1.0, 0.0, 0.0])
        self.curr_vel = np.array([0.0, 0.0, 0.0])

        self.kp = np.array([10, 10, 20])
        self.kd = np.array([4, 4, 5])

        self.viewer = None
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt)) / self.frameskip}

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        # self.t += self.dt
        # print(action)

        action = np.clip(action * self.action_scale, self.action_low, self.action_high)
        self.curr_act = action
        oldPos = self.particleSim.positions.copy()
        self.do_simulation(action, self.frameskip)
        # print(self.curr_pos)
        done = False
        obs = self._get_obs()

        dist_rwd_positive = 0

        reward = np.sum(abs(oldPos)) - np.sum(np.abs(self.particleSim.positions))
        # print(reward)
        return obs, reward, done, {'rwd': reward, 'dist_rwd_positive': dist_rwd_positive,
                                   }

    def generate_template(self):
        template = np.zeros_like(self.density_map)

        # for i in range(len(template)):
        #     for j in range(len(template)):
        #         pos = self.grid_idx_to_pos(i, j)
        #         if (pos[0] < 0):
        #             template[i][j] = 1
        return template

    def _get_obs(self):

        obs = np.concatenate(
            [self.curr_pos[0:2], [np.sin(self.curr_pos[2]), np.cos(self.curr_pos[2])], self.curr_vel[0:2],
             [np.sin(self.curr_vel[2]), np.cos(self.curr_vel[2])],
             self.density_map.flatten() / 5]).ravel()
        return obs

    def first_nonzero(self, arr, axis, invalid_val=-1):
        mask = arr != 0
        return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)

    def do_simulation(self, act, frameskip):
        normalize_targ_vec = act[2::] / np.linalg.norm(act[2::])
        target_ang = np.angle(normalize_targ_vec[0] + normalize_targ_vec[1] * 1j)
        if (target_ang < 0):
            target_ang += 2 * np.pi

        targ_pos = np.zeros(3)
        targ_pos[0:2] = act[0:2]
        targ_pos[2] = target_ang
        tau = self.kp * (targ_pos - self.curr_pos) - self.kd * self.curr_vel
        # print(act)
        r1 = targ_pos[2] - self.curr_pos[2]
        r2 = targ_pos[2] + 2 * np.pi - self.curr_pos[2]
        r3 = targ_pos[2] - 2 * np.pi - self.curr_pos[2]
        rl = [r1, r2, r3]
        r = rl[np.argmin(np.array([abs(r1), abs(r2), abs(r3)]))]

        tau[2] = self.kp[2] * (r) - self.kd[2] * self.curr_vel[2]

        self.curr_vel = self.curr_vel + tau * self.dt
        # print(r)
        self.curr_pos = self.curr_pos + self.curr_vel * self.dt

        if (self.curr_pos[2] < 0):
            self.curr_pos[2] += np.pi * 2
        elif (self.curr_pos[2] > np.pi * 2):
            self.curr_pos[2] -= np.pi * 2

        self.curr_pos = np.clip(self.curr_pos, self.pos_low, self.pos_high)
        x_0 = np.array([(self.curr_pos[0]) + self.half_length * np.cos(self.curr_pos[2]),
                        self.curr_pos[1] + self.half_length * np.sin(self.curr_pos[2])])

        x_1 = np.array([(self.curr_pos[0]) - self.half_length * np.cos(self.curr_pos[2]),
                        self.curr_pos[1] - self.half_length * np.sin(self.curr_pos[2])])

        norm_dir = np.array([x_1[1] - x_0[1], -(x_1[0] - x_0[0])])
        if (np.linalg.norm(norm_dir) > 0.0005):
            norm_dir = norm_dir / np.linalg.norm(norm_dir)

        for fs in range(frameskip):
            self.particleSim.advance(x_0, x_1, norm_dir)
        self.density_map = self.particleSim.fillDensityMap(self.world_size, self.numCells)

    def _reset(self):
        self.curr_act = None
        self.grid_world = [[[] for _ in range(len(self.density_map[0]))] for _ in range(len(self.density_map))]

        rand_angle = np.random.uniform(-np.pi / 3, np.pi / 3, 1)
        if (rand_angle < 0):
            rand_angle += 2 * np.pi
        rand_pos = np.concatenate([np.random.uniform(-0.25, 0.25, 2), rand_angle])

        self.curr_pos = np.array([1, 0, 0]) + rand_pos
        self.curr_vel = np.array([0, 0, 0]) + np.random.uniform(-0.1, 0.1, 3)

        self.particleSim.randomInit(self.world_size)
        self.density_map = self.particleSim.fillDensityMap(self.world_size, self.numCells)

        return self._get_obs()

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 800
        screen_height = 800
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(screen_width, screen_height)

            for i in range(len(self.density_map) + 1):
                left = (self.world_size_view - self.world_size) * 0.5 / self.world_size_view * screen_width
                right = screen_width * self.world_size / self.world_size_view + left
                top = i / float(len(self.density_map)) * screen_height * self.world_size / self.world_size_view + left
                horizontalLine = rendering.Line(start=(left, top), end=(right, top))
                horizontalLine.attrs.pop(-1)
                horizontalLine.add_attr(rendering.LineWidth(3))

                self.viewer.add_geom(horizontalLine)
                self.geomCnt += 1

            for i in range(len(self.density_map[0]) + 1):
                top = (self.world_size_view - self.world_size) * 0.5 / self.world_size_view * screen_height
                bottom = screen_height * self.world_size / self.world_size_view + top
                left = i / float(len(self.density_map[0])) * screen_width * self.world_size / self.world_size_view + top
                vertLine = rendering.Line(start=(left, top), end=(left, bottom))
                vertLine.attrs.pop(-1)
                vertLine.add_attr(rendering.LineWidth(3))

                self.viewer.add_geom(vertLine)
                self.geomCnt += 1

            # for i in range(len(self.template)):
            #     for j in range(len(self.template[0])):
            #         posX, posY = self.grid_idx_to_pos(i, j)
            #         screenX, screenY = self.world_to_screen([posX, posY], screen_width, screen_height)
            #         fill = rendering.make_circle(
            #             radius=(screen_width * self.world_size / self.world_size_view) / (
            #                     2.0 * len(self.template)) - 1)
            #         trans = rendering.Transform((screenX, screenY))
            #         fill.add_attr(trans)
            #         normalized_template = self.template[i][j] / np.max(self.template)
            #         color = (1 - normalized_template, 1, 1 - normalized_template)
            #         fill.set_color(color[0], color[1], color[2])
            #         # if (color[0] != 1):
            #         self.viewer.add_geom(fill)
            #         self.geomCnt += 1

        geoms = self.viewer.geoms[:self.geomCnt]
        # geoms = []

        # for i in range(len(self.density_map)):
        #     for j in range(len(self.density_map[0])):
        #         posX, posY = self.grid_idx_to_pos(i, j)
        #         screenX, screenY = self.world_to_screen([posX, posY], screen_width, screen_height)
        #         fill = rendering.make_circle(
        #             radius=(screen_width * self.world_size / self.world_size_view) / (3.0 * len(self.density_map)))
        #         # fill = rendering.FilledPolygon(v=[screenX])
        #         trans = rendering.Transform((screenX, screenY))
        #         fill.add_attr(trans)
        #         normalized_density = self.density_map[i][j] / np.max(self.density_map)
        #         color = (1 - normalized_density, 1 - normalized_density, 1 - normalized_density)
        #         fill.set_color(color[0], color[1], color[2])
        #         geoms.append(fill)

        if self.curr_act is not None:
            normalize_targ_vec = self.curr_act[2::] / np.linalg.norm(self.curr_act[2::])
            target_ang = np.angle(normalize_targ_vec[0] + normalize_targ_vec[1] * 1j)
            if (target_ang < 0):
                target_ang += 2 * np.pi

            targ_pos = np.zeros(3)
            targ_pos[0:2] = self.curr_act[0:2]
            targ_pos[2] = target_ang
            # x_1 = np.array([self.curr_act[0], self.curr_act[1] + self.half_length])
            # x_2 = np.array([self.curr_act[0], self.curr_act[1] - self.half_length])
            # norm_dir = np.array([-1, 0])

            x_1 = np.array([(targ_pos[0]) + self.half_length * np.cos(targ_pos[2]),
                            targ_pos[1] + self.half_length * np.sin(targ_pos[2])])

            x_2 = np.array([(targ_pos[0]) - self.half_length * np.cos(targ_pos[2]),
                            targ_pos[1] - self.half_length * np.sin(targ_pos[2])])

            norm = np.linalg.norm(np.array([x_2[1] - x_1[1], -(x_2[0] - x_1[0])]))
            norm_dir = np.array([x_2[1] - x_1[1], -(x_2[0] - x_1[0])]) / norm

            # if (np.linalg.norm(norm_dir) > 0.0005):
            #     norm_dir = norm_dir / np.linalg.norm(norm_dir)

            x_3 = x_2 + 0.3 * norm_dir  # * self.curr_act[4]
            x_4 = x_1 + 0.3 * norm_dir  # * self.curr_act[4]
            screenX_1, screenY_1 = self.world_to_screen(x_1, screen_width, screen_height)
            screenX_2, screenY_2 = self.world_to_screen(x_2, screen_width, screen_height)
            screenX_3, screenY_3 = self.world_to_screen(x_3, screen_width, screen_height)
            screenX_4, screenY_4 = self.world_to_screen(x_4, screen_width, screen_height)
            #
            line = rendering.Line((screenX_1, screenY_1), (screenX_2, screenY_2))
            line.attrs.pop(-1)
            line.add_attr(rendering.LineWidth(10))
            line.set_color(0.3, 0, 0)

            rect = rendering.FilledPolygon(
                [(screenX_1, screenY_1), (screenX_2, screenY_2), (screenX_3, screenY_3), (screenX_4, screenY_4)])
            rect.set_color(0, 0.3, 0.3)

            # geoms.pop(-1)
            # geoms.pop(-1)
            geoms.append(line)
            geoms.append(rect)

            x_1 = np.array([(self.curr_pos[0]) + self.half_length * np.cos(self.curr_pos[2]),
                            self.curr_pos[1] + self.half_length * np.sin(self.curr_pos[2])])

            x_2 = np.array([(self.curr_pos[0]) - self.half_length * np.cos(self.curr_pos[2]),
                            self.curr_pos[1] - self.half_length * np.sin(self.curr_pos[2])])

            norm = np.linalg.norm(np.array([x_2[1] - x_1[1], -(x_2[0] - x_1[0])]))
            norm_dir = np.array([x_2[1] - x_1[1], -(x_2[0] - x_1[0])]) / norm

            # if (np.linalg.norm(norm_dir) > 0.0005):
            #     norm_dir = norm_dir / np.linalg.norm(norm_dir)

            x_3 = x_2 + 0.3 * norm_dir  # * self.curr_act[4]
            x_4 = x_1 + 0.3 * norm_dir  # * self.curr_act[4]
            screenX_1, screenY_1 = self.world_to_screen(x_1, screen_width, screen_height)
            screenX_2, screenY_2 = self.world_to_screen(x_2, screen_width, screen_height)
            screenX_3, screenY_3 = self.world_to_screen(x_3, screen_width, screen_height)
            screenX_4, screenY_4 = self.world_to_screen(x_4, screen_width, screen_height)
            #
            line = rendering.Line((screenX_1, screenY_1), (screenX_2, screenY_2))
            line.attrs.pop(-1)
            line.add_attr(rendering.LineWidth(10))
            line.set_color(1, 0, 0)

            rect = rendering.FilledPolygon(
                [(screenX_1, screenY_1), (screenX_2, screenY_2), (screenX_3, screenY_3), (screenX_4, screenY_4)])
            rect.set_color(0, 1, 1)

            # geoms.pop(-1)
            # geoms.pop(-1)
            geoms.append(line)
            geoms.append(rect)
        else:
            line = rendering.Line((0, 0), (0, 0))
            geoms.append(line)
            geoms.append(line)

        for i in range(len(self.particleSim.positions)):
            pos = self.particleSim.positions[i]
            screenX, screenY = self.world_to_screen(pos, screen_width, screen_height)
            circle = rendering.make_circle(radius=2, res=10)
            trans = rendering.Transform((screenX, screenY))
            circle.add_attr(trans)
            geoms.append(circle)
        self.viewer.geoms = geoms
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def world_to_screen(self, pos, screen_width, screen_height):
        # screenX = pos[0] / self.world_size * screen_width + 0.5 * screen_width
        #         # screenY = pos[1] / self.world_size * screen_height + 0.5 * screen_height
        screenX = pos[0] / self.world_size_view * screen_width + 0.5 * screen_width
        screenY = pos[1] / self.world_size_view * screen_height + 0.5 * screen_height
        return screenX, screenY

    def pos_to_grid_idx(self, pos):

        x_idx = ((pos[0] + self.world_size / 2) / self.cellSize)
        y_idx = ((self.world_size / 2 - pos[1]) / self.cellSize)

        if (x_idx < 0):
            x_idx = int(x_idx) - 1
        else:
            x_idx = int(x_idx)

        if (y_idx < 0):
            y_idx = int(y_idx) - 1
        else:
            y_idx = int(y_idx)

        return y_idx, x_idx

    def grid_idx_to_pos(self, i, j):
        x = j * self.cellSize + 0.5 * self.cellSize - self.world_size / 2
        y = self.world_size / 2 - (i * self.cellSize + 0.5 * self.cellSize)
        return np.array([x, y])
