from MOSEAC import MOSEACAgent
from ReplayBuffer import RandomBuffer
from Adapter import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from ReplayBuffer import device

import numpy as np
import gymnasium
import torch
import os
import shutil
import argparse

from gymnasium.envs.registration import register
from scipy.stats import linregress

import time


def str2bool(v):
    # transfer str to bool for argparse
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def is_downtrend(data):
    """
    Analyze the trend of a given numpy array of data using linear regression.
    Returns True if there is a downtrend (negative slope).

    Args:
    data (np.array): Numpy array of numerical data.

    Returns:
    bool: True if the data shows a downtrend, False otherwise.
    """
    # Generate indices for the data
    indices = np.arange(len(data))

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(indices, data)

    # Check if the slope is negative (indicating a downtrend)
    return slope < 0


def load_checkpoint_path():
    current_path = os.getcwd()
    checkpoint_path = current_path + '/check_points/checkpoint.pth.tar'
    return checkpoint_path


'''Hyper Parameters Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--Load_checkpoint', type=str2bool, default=False, help='Load checkpoint or Not')
parser.add_argument('--ModelIdex', type=int, default=2250000, help='which model to load')
parser.add_argument('--seed', type=int, default=1995, help='seed for training')

parser.add_argument('--total_steps', type=int, default=int(8e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(1e4), help='Model evaluating interval, in steps.')
parser.add_argument('--eval_turn', type=int, default=5, help='Model evaluating times, in episode.')
parser.add_argument('--update_every', type=int, default=50, help='Training Frequency, in steps')
parser.add_argument('--gamma', type=float, default=0.95, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')

parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
# Set it True to enable the SAC V2

parser.add_argument('--fixed_freq', type=float, default=0.0, help='if 0.0, not use fixed frequency')
parser.add_argument('--obs_freq', type=float, default=20.0, help='fixed obs frequency setting by user, should not be 0')
parser.add_argument('--min_time', type=float, default=0.02, help='min time of taking one action, should not be 0')
parser.add_argument('--max_time', type=float, default=0.5, help='max time of taking one action, should not be 0')
parser.add_argument('--ep_max_length', type=int, default=500, help='maximum step for one training episode')
parser.add_argument('--adaptive_gain', type=str2bool, default=True, help='Use the adaptive elastic parameters')
parser.add_argument('--increase_amount', type=float, default=1e-4, help='monotony increasing amount')

parser.add_argument('--time_benchmark', type=str2bool, default=False, help='bench mark the time cost')


opt = parser.parse_args()
print(opt)
print(device)


def evaluate_policy(env, model, render, max_time, max_action_s, min_time, obs_freq, fixed_freq, world_size):
    scores = 0
    total_time = 0
    total_energy = 0
    turns = opt.eval_turn
    for j in range(turns):
        current_step_eval = 0
        ep_r = 0
        dead = False
        obs, info = env.reset()
        agent_obs = obs['agent_pos']
        goal = obs['target']
        linear_speed = obs['linear_speed']
        yaw_speed = obs['yaw']
        time_last_step = obs['time']
        control = obs['control']
        radar = obs['radar']
        agent_obs_uniform = agent_obs / world_size
        goal_uniform = goal / world_size
        linear_speed_uniform = linear_speed / max_action_s
        yaw_uniform = yaw_speed / max_action_s
        time_last_step_uniform = (time_last_step - min_time) / (max_time - min_time)
        control_uniform = control / max_action_s
        radar_uniform = radar / world_size
        s = np.concatenate([agent_obs_uniform, goal_uniform, linear_speed_uniform, yaw_uniform,
                           time_last_step_uniform, control_uniform, radar_uniform], axis=0)

        time_epoch = 0
        while not dead:
            current_step_eval += 1
            a = model.select_action(s, deterministic=True, with_logprob=False)
            if fixed_freq:
                a_s_eval = a[0:2]
                act_s_eval = Action_adapter(a_s_eval, max_action_s)
                act_t_eval = np.array([1.0 / obs_freq])
            else:
                a_t_eval = a[0]
                a_s_eval = a[1:3]
                act_s_eval = Action_adapter(a_s_eval, max_action_s)
                act_t_eval = Action_t_relu6_adapter(a_t_eval, min_time, max_time)
                act_t_eval = np.array([act_t_eval])
            act = np.concatenate([act_t_eval, act_s_eval], axis=0)
            obs, r, terminated, truncated, info = env.step(act)
            agent_obs = obs['agent_pos']
            goal = obs['target']
            linear_speed = obs['linear_speed']
            yaw_speed = obs['yaw']
            time_last_step = obs['time']
            control = obs['control']
            radar = obs['radar']
            agent_obs_uniform = agent_obs / world_size
            goal_uniform = goal / world_size
            linear_speed_uniform = linear_speed / max_action_s
            yaw_uniform = yaw_speed / max_action_s
            time_last_step_uniform = (time_last_step - min_time) / (max_time - min_time)
            control_uniform = control / max_action_s
            radar_uniform = radar / world_size
            s_prime = np.concatenate([agent_obs_uniform, goal_uniform, linear_speed_uniform, yaw_uniform,
                                     time_last_step_uniform, control_uniform, radar_uniform], axis=0)
            s = s_prime
            time_epoch += act_t_eval
            if terminated or truncated:
                dead = True
            ep_r += r
            if render:
                env.render()
        energy = current_step_eval
        total_energy += energy
        scores += ep_r
        total_time += time_epoch
    return float(scores / turns), float(total_time / turns), float(total_energy / turns)


def main():
    global t
    increase_amount = opt.increase_amount
    adaptive_gain = opt.adaptive_gain
    write = opt.write  # Use SummaryWriter to record the training.
    render = opt.render
    seed = opt.seed
    env_with_dead = True
    steps_per_epoch = opt.ep_max_length
    register(
        id="limo_world",
        entry_point="envs:LimoEnv",
        max_episode_steps=steps_per_epoch,
    )
    env = gymnasium.make('limo_world')
    fixed_freq = opt.fixed_freq
    state_dim = 49
    if fixed_freq:
        action_dim = 2
    else:
        action_dim = 3
    min_time = opt.min_time
    max_time = opt.max_time
    max_action_s = 1.0

    world_size = 1.5
    time_benchmark = opt.time_benchmark
    obs_freq = opt.obs_freq

    # Interaction config:
    start_steps = 5 * steps_per_epoch  # in steps
    update_after = 2 * steps_per_epoch  # in steps
    update_every = opt.update_every
    total_steps = opt.total_steps
    eval_interval = opt.eval_interval  # in steps
    save_interval = opt.save_interval  # in steps

    load_checkpoint = opt.Load_checkpoint
    time_now = str(datetime.now())[0:-10]
    time_now = ' ' + time_now[0:13] + '_' + time_now[-2::]
    save_path = '/model/SEAC_time{}'.format("limo_world") + time_now
    load_path = '/model/'
    dir_path = os.getcwd()
    save_path = dir_path + save_path
    load_path = dir_path + load_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # SummaryWriter config:
    if write:
        write_path = 'runs/SEAC_time{}'.format("limo") + time_now
        if os.path.exists(write_path):
            shutil.rmtree(write_path)
        writer = SummaryWriter(log_dir=write_path)
    else:
        writer = None
    # Model hyperparameter config:
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "gamma": opt.gamma,
        "hid_shape": (opt.net_width, opt.net_width, opt.net_width),
        "a_lr": opt.a_lr,
        "c_lr": opt.c_lr,
        "batch_size": opt.batch_size,
        "alpha": opt.alpha,
        "adaptive_alpha": opt.adaptive_alpha,
        "save": save_path,
        "load": load_path
    }

    model = MOSEACAgent(**kwargs)
    if adaptive_gain:
        reward_buffer = QueueUpdater(c=int(opt.ep_max_length))
        reward_buffer.reset()
        reward_average_buffer = QueueUpdater(c=int(update_every))
    else:
        reward_buffer = None
        reward_average_buffer = None
    if not os.path.exists('model'):
        os.mkdir('model')
    if opt.Loadmodel:
        model.load(opt.ModelIdex)

    replay_buffer = RandomBuffer(state_dim, action_dim, env_with_dead, max_size=int(1e5))

    current_steps = 0
    obs, info = env.reset()
    agent_obs = obs['agent_pos']
    goal = obs['target']
    linear_speed = obs['linear_speed']
    yaw_speed = obs['yaw']
    time_last_step = obs['time']
    control = obs['control']
    radar = obs['radar']
    agent_obs_uniform = agent_obs / world_size
    goal_uniform = goal / world_size
    linear_speed_uniform = linear_speed / max_action_s
    yaw_uniform = yaw_speed / max_action_s
    time_last_step_uniform = (time_last_step - min_time) / (max_time - min_time)
    control_uniform = control / max_action_s
    radar_uniform = radar / world_size
    s = np.concatenate([agent_obs_uniform, goal_uniform, linear_speed_uniform, yaw_uniform,
                        time_last_step_uniform, control_uniform, radar_uniform], axis=0)

    fixed_freq = np.array([fixed_freq])
    tricker = 0
    time_old = 0.0
    checkpoint_path = load_checkpoint_path()
    if load_checkpoint:
        if os.path.isfile(checkpoint_path):
            print("Try to load checkpoint")
            t = model.load_checkpoint(checkpoint_path)
            total_steps = total_steps - t
        else:
            t = 0
    else:
        print("Don't load checkpoint")
        t = 0
    print("start from epoch:", t)
    try:
        for t in range(total_steps):
            current_steps += 1
            if render:
                env.render()
            if t < start_steps:
                act = env.action_space.sample()
                act_t = act[0]
                act_s = act[1:3]
                act_t = Act_t_correction(act_t)  # to make sure that the time should be positive
                act_t = (max_time - min_time) * (act_t / max_time) + min_time
                # fixed the range of time from [0,-0.1] to [0.02, 1.0]
                act_t = np.array([act_t])
                if fixed_freq:
                    act_t = np.array([1.0/obs_freq])
                act = np.concatenate([act_t, act_s], axis=0)
                a_s = Action_adapter_reverse(act_s, max_action_s)
                a_t = Action_t_relu6_adapter_reverse(act_t, min_time, max_time)
                if fixed_freq:
                    a = a_s
                else:
                    a = np.concatenate([a_t, a_s], axis=0)
            else:
                a = model.select_action(s, deterministic=False, with_logprob=False)
                if fixed_freq:
                    a_s = a[0:2]
                    act_s = Action_adapter(a_s, max_action_s)
                    act_t = np.array([1.0 / obs_freq])
                else:
                    a_s = a[1:3]
                    a_t = a[0]
                    act_s = Action_adapter(a_s, max_action_s)
                    act_t = Action_t_relu6_adapter(a_t, min_time, max_time)
                    act_t = np.array([act_t])
                act = np.concatenate([act_t, act_s], axis=0)
            obs, rew, terminated, truncated, info = env.step(act)
            agent_obs = obs['agent_pos']
            goal = obs['target']
            linear_speed = obs['linear_speed']
            yaw_speed = obs['yaw']
            time_last_step = obs['time']
            control = obs['control']
            radar = obs['radar']
            agent_obs_uniform = agent_obs / world_size
            goal_uniform = goal / world_size
            linear_speed_uniform = linear_speed / max_action_s
            yaw_uniform = yaw_speed / max_action_s
            time_last_step_uniform = (time_last_step - min_time) / (max_time - min_time)
            control_uniform = control / max_action_s
            radar_uniform = radar / world_size
            s_prime = np.concatenate([agent_obs_uniform, goal_uniform, linear_speed_uniform, yaw_uniform,
                                     time_last_step_uniform, control_uniform, radar_uniform], axis=0)
            s_prime_t = torch.tensor(np.float32(s_prime))
            if adaptive_gain:
                reward_buffer.append(rew)
            if terminated or truncated:
                dead = True
                if adaptive_gain:
                    reward_average_buffer.append(np.mean(reward_buffer.to_numpy()))
                    reward_buffer.reset()
            else:
                dead = False
            s_t = torch.tensor(np.float32(s))
            a_t = torch.tensor(a)
            replay_buffer.add(s_t, a_t, rew, s_prime_t, dead)
            s = s_prime
            if (t+1) % 2000 == 0:
                print('CurrentPercent:', ((t + 1)*100.0)/total_steps, '%')
                if tricker == 0:
                    time_old = time.time()
                else:
                    time_new = time.time()
                    time_diff = time_new - time_old
                    if time_benchmark:
                        print("Predicted Completion Time：", time_diff * ((total_steps-t)/500), "in seconds")
                    time_old = time_new
                tricker += 1

            # 50 environment steps company with 50 gradient steps.
            # Stabler than 1 environment step company with 1 gradient step.
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    model.train(replay_buffer)
            if adaptive_gain:
                reward_average_determine = reward_average_buffer.to_numpy()
                if is_downtrend(reward_average_determine):
                    env.unwrapped.update_gain_r(increase_amount)
                reward_average_buffer.reset()

            '''save model'''
            if (t + 1) % save_interval == 0:
                model.save(t + 1)

            '''record & log'''
            if (t + 1) % eval_interval == 0:
                score, average_time, average_energy_cost = evaluate_policy(env, model, False, max_time, max_action_s,
                                                                           min_time, obs_freq, fixed_freq, world_size)
                if write:
                    writer.add_scalar('ep_r', score, global_step=t + 1)
                    writer.add_scalar('alpha', model.alpha, global_step=t + 1)
                    writer.add_scalar('average_time', average_time, global_step=t + 1)
                    writer.add_scalar('average_energy_cost', average_energy_cost, global_step=t + 1)
                print('EnvName: limo_world', 'TotalSteps:', t + 1, 'score:', score, 'average_time:', average_time,
                      'average_energy_cost:', average_energy_cost)
            if dead:
                current_steps = 0
                obs, info = env.reset()
                agent_obs = obs['agent_pos']
                goal = obs['target']
                linear_speed = obs['linear_speed']
                yaw_speed = obs['yaw']
                time_last_step = obs['time']
                control = obs['control']
                radar = obs['radar']
                agent_obs_uniform = agent_obs / world_size
                goal_uniform = goal / world_size
                linear_speed_uniform = linear_speed / max_action_s
                yaw_uniform = yaw_speed / max_action_s
                time_last_step_uniform = (time_last_step - min_time) / (max_time - min_time)
                control_uniform = control / max_action_s
                radar_uniform = radar / world_size
                s = np.concatenate([agent_obs_uniform, goal_uniform, linear_speed_uniform, yaw_uniform,
                                    time_last_step_uniform, control_uniform, radar_uniform], axis=0)
    except KeyboardInterrupt:
        # 当检测到Ctrl+C时，保存当前的检查点然后退出
        model.save_checkpoint(t, checkpoint_path)
        print("Training interrupted. Checkpoint saved.")
        writer.close()
        env.close()
        exit(0)

    writer.close()
    env.close()


if __name__ == '__main__':
    main()
