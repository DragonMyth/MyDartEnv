import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from gym.envs.flex import flex_env
import pygame as pg
import itertools
from pygame.locals import *

try:
    import bindings as pyFlex
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: PyFlex Binding is not installed correctly)".format(e))

def generate_manual_action(w, a, s, d, cw, ccw, ghost, skip, obs):
    bar_state = obs[0, 0:4]

    act = np.zeros((1, 4))

    linear_scale = 2
    ang_scale = 0.1 * np.pi
    linear_relative_target = np.zeros(2)
    target_angle = 0
    ghost_cont = 0
    if w:
        linear_relative_target += (np.array([0, -1]) * linear_scale)

    if s:
        linear_relative_target += (np.array([0, 1]) * linear_scale)

    if a:
        linear_relative_target += (np.array([-1, 0]) * linear_scale)
    if d:
        linear_relative_target += (np.array([1, 0]) * linear_scale)

    if ghost:
        ghost_cont = 1
    if ccw:
        target_angle = 1 * ang_scale
    if cw:
        target_angle = -1 * ang_scale
    if not skip:
        rot_vec = np.ones((2, 2))

        rot_vec[0, 0] = np.cos(target_angle)
        rot_vec[0, 1] = -np.sin(target_angle)
        rot_vec[1, 0] = np.sin(target_angle)
        rot_vec[1, 1] = np.cos(target_angle)

        rot_vec = np.matmul(bar_state[2::].transpose(), rot_vec.transpose()).transpose()
        act[0, 0:2] =  linear_relative_target
        # act[0, 2:4] = rot_vec
        # act[0, 4] = ghost_cont
        act[0,2]  = target_angle
        act[0,3] = ghost_cont
    else:
        # act[0, 0:4] = bar_state
        # act[0, 4] = 0
        pass

    # act = np.array([[0.52088761, -2.95443344, 1, 0, 0]])
    # act = np.array([[1,0 ,1, 0, 0]])

    # print(act)
    return act

def generate_manual_action_abs_rot(w, a, s, d, cw, ccw, ghost, skip, obs):
    bar_state = obs[0, 0:4]

    act = np.zeros((1, 5))

    linear_scale = 2
    ang_scale = 0.1 * np.pi
    linear_relative_target = np.zeros(2)

    target_angle =0# np.arctan2(bar_state[1,1],bar_state[1,0])
    ghost_cont = 0
    if w:
        linear_relative_target += (np.array([0, -1]) * linear_scale)

    if s:
        linear_relative_target += (np.array([0, 1]) * linear_scale)

    if a:
        linear_relative_target += (np.array([-1, 0]) * linear_scale)
    if d:
        linear_relative_target += (np.array([1, 0]) * linear_scale)

    if ghost:
        ghost_cont = 1
    if ccw:
        target_angle += 1 * ang_scale
    if cw:
        target_angle += -1 * ang_scale
    if not skip:
        rot_vec = np.ones((2, 2))

        rot_vec[0, 0] = np.cos(target_angle)
        rot_vec[0, 1] = -np.sin(target_angle)
        rot_vec[1, 0] = np.sin(target_angle)
        rot_vec[1, 1] = np.cos(target_angle)

        rot_vec = np.matmul(bar_state[1].transpose(), rot_vec.transpose()).transpose()
        act[0, 0:2] =  linear_relative_target
        act[0, 2:4] = rot_vec
        act[0, 4] = ghost_cont

    else:
        # act[0, 0:4] = bar_state
        # act[0, 4] = 0
        pass

    # act = np.array([[0.52088761, -2.95443344, 1, 0, 0]])
    # act = np.array([[1,0 ,1, 0, 0]])

    # print(act)
    return act


if __name__ == '__main__':
    import gym
    env = gym.make("FlexPlasticReshaping-v0")

    env.save_video = True
    env.video_path = 'manual_control_data'
    import os

    if not os.path.exists(env.unwrapped.video_path):
        os.makedirs(env.unwrapped.video_path)

    obs = env.reset()
    cnt = 0
    paused = True

    all_info = []
    ret = 0
    while cnt < 500:

        act = np.zeros((1, 5))

        events = pg.event.get()

        W = False
        A = False
        S = False
        D = False
        CW = False
        CCW = False
        Ghost = False
        skip = False
        keys = pg.key.get_pressed()

        if keys[pg.K_r]:
            paused = False
        if keys[pg.K_p]:
            paused = True
        if keys[pg.K_w]:
            W = True
        if keys[pg.K_a]:
            A = True
        if keys[pg.K_s]:
            S = True
        if keys[pg.K_d]:
            D = True

        if keys[pg.K_j]:
            CCW = True

        if keys[pg.K_k]:
            CW = True

        if keys[pg.K_SPACE]:
            skip = True

        if keys[pg.K_LSHIFT]:
            Ghost = True
        key_pressed = W or A or S or D or CW or CCW or Ghost or skip
        if (key_pressed or not paused):
            env.render()
            state = env.unwrapped.get_state()

            act = generate_manual_action_abs_rot(W, A, S, D, CW, CCW, Ghost, skip, state)
            # print(act)
            act = act[:]/env.unwrapped.action_scale
            obs, rwd, done, info = env.step(act)
            ret+=rwd[0]
            all_info.append(info)
            # state = env.get_full_state()
            #
            # states.append(state[0, 4::].flatten())
            cnt += 1

            if done:
                break

    import matplotlib.pyplot as plot

    iters = np.arange(1, len(all_info) + 1, 1)
    data_list = {}
    for i in range(len(all_info)):

        for key in all_info[0]:
            if key not in data_list:
                data_list[key] = []
            data_list[key].append(all_info[i][key])

    # total_ret = np.sum(data_list['rwd'])
    # print("Total return of the trajectory is: ", total_ret)

    # sns.set_palette('hls', len(data_list))
    for key in sorted(data_list.keys()):
        # print(key)
        if (key != 'actions' and key != 'states'):
            cnt += 1
            plot.plot(iters, data_list[key],
                      label=str(key))
            plot.yscale('symlog')

    plot.xlabel('Time Steps')
    plot.ylabel('Step Reward')
    plot.legend()
    plot.show()
    print("Total Return: ",ret)
    # np.savetxt("sample_data.csv",states,delimiter=",")
    # env = PlasticSpringReshapingEnvManualControl()
    #
    # obs = env.reset()
    # for _ in range(1000):
    # env.render()
    # act = np.zeros((1, 5))
    # obs, rwd, done, info = env.step(act)
