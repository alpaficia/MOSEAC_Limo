from MOSEAC import MOSEACAgent
from ReplayBuffer import RandomBuffer
from Action_time_buffer import ActionTimeBuffer
from Adapter import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from ReplayBuffer import device

import numpy as np
import torch
import os
import shutil
import argparse
import time
import warnings
warnings.simplefilter('ignore', np.RankWarning)

os.environ['NUMEXPR_MAX_THREADS'] = '16' 
# You may changed the Threads based on your PC
from tmrl import get_environment


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

'''Hyper Parameters Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=2250000, help='which model to load')

parser.add_argument('--total_steps', type=int, default=int(8e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(5e4), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(5e4), help='Model evaluating interval, in steps.')
parser.add_argument('--eval_turn', type=int, default=3, help='Model evaluating times, in episode.')
parser.add_argument('--update_every', type=int, default=20, help='Training Frequency, in steps')
parser.add_argument('--gamma', type=float, default=0.95, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width')
parser.add_argument('--a_lr', type=float, default=2e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')

parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
parser.add_argument('--alpha', type=float, default=0.01, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=False, help='Use adaptive_alpha or Not')
# Set it True to enable the SAC V2

parser.add_argument('--min_time', type=float, default=(1.0/30.0), help='min time of taking one action, should not be 0')
parser.add_argument('--max_time', type=float, default=(1.0/5.0), help='max time of taking one action, should not be unlimited')

parser.add_argument('--alpha_m', type=float, default=1.0, help='reward parameters for accomplishing the task')
parser.add_argument('--alpha_max', type=float, default=10.0, help='reward parameters for energy cost')
parser.add_argument('--psi', type=float, default=1e-4, help='adaptive parameter to adjust alpha_m')

opt = parser.parse_args()
print(opt)
print(device)
atbuffer = ActionTimeBuffer()


def evaluate_policy(env, model, max_time, min_time, max_action_m, alpha_m, alpha_epsilon):
    scores = 0
    total_time = 0
    total_energy = 0
    turns = opt.eval_turn
    for j in range(turns):
        atbuffer.append(0.05)
        current_step_eval = 0
        ep_r = 0
        dead = False
        obs, info = env.reset()
        speed = obs[0]
        rpm = obs[1]
        gear = obs[2]
        image = obs[3]  # shape: (1, 64, 64, 3) in rgb
        history_action_0 = obs[4]  # history action value
        history_action_1 = obs[5]
        atbuffer.reset()
        init_time = atbuffer.to_numpy()  # init control frequency
        s = np.concatenate([speed, rpm, gear, np.array([current_step_eval]), init_time, history_action_0, history_action_1], axis=0)
        time_epoch = 0
        while not dead:
            current_step_eval += 1
            # Take deterministic actions at test time
            a = model.select_action(s, image, deterministic=True, with_logprob=False)
            a_t_eval = a[0]
            a_m_eval = a[1:]
            act_m_eval = Action_adapter(a_m_eval, max_action_m)
            act_t_eval = Action_t_relu6_adapter(a_t_eval, max_time)
            if act_t_eval <= min_time:
                act_t_eval = min_time
            act_t_eval = np.array([act_t_eval])
            env.unwrapped.set_time_step_duration(time_step_duration=float(act_t_eval))
            env.unwrapped.set_start_obs_capture(start_obs_capture=float(act_t_eval-0.01))
            atbuffer.append(float(act_t_eval))
            obs, r, terminated, truncated, info = env.step(act_m_eval)
            reward = reward_adapter(r, alpha_m, alpha_epsilon, min_time, act_t_eval)
            speed = obs[0]
            rpm = obs[1]
            gear = obs[2]
            image_prime = obs[3]  # shape: (1, 64, 64) in gray
            history_action_0 = obs[4]
            history_action_1 = obs[5]
            action_time = atbuffer.to_numpy()
            s_prime = np.concatenate([speed, rpm, gear, np.array([current_step_eval]), action_time, history_action_0, history_action_1], axis=0)
            s = s_prime
            image = image_prime
            time_epoch += act_t_eval
            if terminated or truncated:
                dead = True
            ep_r += reward
        energy = current_step_eval
        total_energy += energy
        scores += ep_r
        total_time += time_epoch
    return float(scores / turns), float(total_time / turns), float(total_energy / turns)


def main():
    write = opt.write  # Use SummaryWriter to record the training.
    env_with_dead = True
    env = get_environment()  # load Trackmania env, you need to activate TM23 window
    time.sleep(1.0)  # just so we have time to focus the TM23 window after starting the script
    state_dim = 13
    action_dim = 4 
    img_shape = (64, 64)
    max_action_m = 1.0
    min_time = opt.min_time
    max_time = opt.max_time
    
    alpha_m = opt.alpha_m
    alpha_m, alpha_epsilon = update_gain_r(alpha_m, 0.0)
    psi = opt.psi
    alpha_max = opt.alpha_max

    # Interaction config:
    start_steps = 5000  # in steps
    update_after = 2000  # in steps
    update_every = opt.update_every
    total_steps = opt.total_steps
    eval_interval = opt.eval_interval  # in steps
    save_interval = opt.save_interval  # in steps
    
    reward_buffer = QueueUpdater(c=int(2000))  # change it while you change the maximum step of one epsoide.
    reward_buffer.reset()
    reward_average_buffer = QueueUpdater(c=int(update_every))
    reward_average_buffer.reset()

    # SummaryWriter config:
    if write:
        time_now = str(datetime.now())[0:-10]
        time_now = ' ' + time_now[0:13] + '_' + time_now[-2::]
        write_path = 'runs/MOSEAC_time{}'.format("TMRL_Trackmania") + time_now
        if os.path.exists(write_path):
            shutil.rmtree(write_path)
        writer = SummaryWriter(log_dir=write_path)
    else:
        writer = None

    # Model hyperparameter config:
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "image_shape": img_shape,
        "gamma": opt.gamma,
        "hid_shape": (opt.net_width, opt.net_width),
        "a_lr": opt.a_lr,
        "c_lr": opt.c_lr,
        "batch_size": opt.batch_size,
        "alpha": opt.alpha,
        "adaptive_alpha": opt.adaptive_alpha
    }

    model = MOSEACAgent(**kwargs)
    if not os.path.exists('model'):
        os.mkdir('model')
    if opt.Loadmodel:
        model.load(opt.ModelIdex)

    replay_buffer = RandomBuffer(state_dim, action_dim, env_with_dead, max_size=int(1e5))
    current_steps = 0
    train_t_history = 0
    number_of_eval = 1
    numbers_of_save = 1
    obs, info = env.reset()
    speed = obs[0]
    rpm = obs[1]
    gear = obs[2]
    image = obs[3]  # shape: (4, 64, 64) in gray image.
    history_action_0 = obs[4]  # history action value
    history_action_1 = obs[5]  # history action value
    atbuffer.reset()
    init_time = atbuffer.to_numpy()
    s = np.concatenate([speed, rpm, gear, np.array([current_steps]), init_time, history_action_0, history_action_1], axis=0)
    for t in range(total_steps):
        current_steps += 1
        atbuffer.append(0.05)
        if t < start_steps:
            # Random explore for start_steps, but first 10 step with certainty moving speed
            act_m = env.action_space.sample()
            act_t = np.random.uniform(min_time, max_time, 1)
            a_m = Action_adapter_reverse(act_m, max_action_m)
            a_t = Action_t_relu6_adapter_reverse(act_t, max_time)
            a = np.concatenate([a_t, a_m], axis=0)
        else:
            a = model.select_action(s, image, deterministic=False, with_logprob=False)
            a_m = a[1:]
            a_t = a[0]
            act_m = Action_adapter(a_m, max_action_m)
            act_t = Action_t_relu6_adapter(a_t, max_time)
            if act_t <= min_time:
                act_t = min_time  # We don't want the time goes to 0, which makes many troubles
            act_t = np.array([act_t])
        env.unwrapped.set_time_step_duration(time_step_duration=float(act_t))
        env.unwrapped.set_start_obs_capture(start_obs_capture=float(act_t))
        atbuffer.append(float(act_t))
        obs, rew, terminated, truncated, info = env.step(act_m)
        reward_buffer.append(rew)
        
        reward = reward_adapter(rew, alpha_m, alpha_epsilon, min_time, act_t)
        speed = obs[0]
        rpm = obs[1]
        gear = obs[2]
        image_prime = obs[3]  # shape: (4, 64, 64, 3) in rgb
        history_action_0 = obs[4]
        history_action_1 = obs[5]
        action_time = atbuffer.to_numpy()
        s_prime = np.concatenate([speed, rpm, gear, np.array([current_steps]), action_time, history_action_0, history_action_1], axis=0)
        s_prime_t = torch.tensor(np.float32(s_prime))
        if terminated or truncated:
            dead = True
            reward_average_buffer.append(np.mean(reward_buffer.to_numpy()))
            reward_buffer.reset()
        else:
            dead = False
        s_t = torch.tensor(np.float32(s))
        a_t = torch.tensor(a)
        image_t = torch.tensor(image)
        image_prime_t = torch.tensor(image_prime)
        replay_buffer.add(s_t, image_t, a_t, reward, s_prime_t, image_prime_t, dead)
        s = s_prime
        image = image_prime
        if (t+1) % 5000 == 0:
            print('CurrentPercent:', ((t + 1)*100.0)/total_steps, '%')

        if dead:
            # 50 environment steps company with 50 gradient steps.
            # Stabler than 1 environment step company with 1 gradient step.
            if t >= update_after and t >= (train_t_history * update_every) + update_after:
                for j in range(update_every):
                    model.train(replay_buffer)
                train_t_history += 1
            reward_average_determine = reward_average_buffer.to_numpy()
            if is_downtrend(reward_average_determine):
                alpha_m, alpha_epsilon = update_gain_r(alpha_m, psi)
                if alpha_m >= alpha_max:
                    alpha_m = alpha_max
                reward_average_buffer.reset()
            
            '''record & log'''
            if t >= number_of_eval * eval_interval:
                score, average_time, average_energy_cost = evaluate_policy(env, model, max_time, min_time, max_action_m, 
                                                                           alpha_m, alpha_epsilon)
                if write:
                    writer.add_scalar('ep_r', score, global_step=t + 1)
                    writer.add_scalar('alpha', model.alpha, global_step=t + 1)
                    writer.add_scalar('average_time', average_time, global_step=t + 1)
                    writer.add_scalar('average_energy_cost', average_energy_cost, global_step=t + 1)
                print('EnvName: TMRL_Trackmania', 'TotalSteps:', t + 1, 'score:', score, 'average_time:', average_time,
                      'average_energy_cost:', average_energy_cost)
                number_of_eval += 1
            
            '''save model'''
            if t >= numbers_of_save * save_interval:
                model.save(t)
                numbers_of_save += 1
            
            current_steps = 0
            obs, info = env.reset()
            speed = obs[0]
            rpm = obs[1]
            gear = obs[2]
            image = obs[3]  # shape: (1, 64, 64, 3) in rgb
            history_action_0 = obs[4]  # history action value
            history_action_1 = obs[5]  # history action value
            atbuffer.reset()
            init_time = atbuffer.to_numpy()  # init control frequency
            s = np.concatenate([speed, rpm, gear, np.array([current_steps]), init_time, history_action_0, history_action_1], axis=0)

    writer.close()
    env.close()


if __name__ == '__main__':
    main()
