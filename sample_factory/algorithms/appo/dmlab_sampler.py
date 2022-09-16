from logging import exception
import sys
import time
from collections import deque

import numpy as np
import torch
import os
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.action_distributions import ContinuousActionDistribution
from sample_factory.algorithms.utils.algo_utils import ExperimentStatus
from sample_factory.algorithms.utils.arguments import parse_args, load_from_checkpoint
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict
import h5py
import json
import random


def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'rewards': [],
            }

def append_data(data, obs, act, rew, done):
    len_check = {len(obs), len(act), len(rew), len(done)}
    if len(len_check) == 1:
        data['observations'].extend(obs)
        data['actions'].extend(act)
        data['rewards'].extend(rew)
        data['terminals'].extend(done)
        return True
    else:
        log.info('traj error, deserted')
        return False

def merge_data(data1, data2):
    data1['observations'].extend(data2['observations'])
    data1['actions'].extend(data2['actions'])
    data1['rewards'].extend(data2['rewards'])
    data1['terminals'].extend(data2['terminals'])

def npify(data):
    for k in data:
        if k in ['terminals', 'timeouts']:
            dtype = np.bool_
            data[k] = np.array(data[k], dtype=dtype)
        else:
            data[k] = np.array(data[k])

def enjoy(cfg):
    cfg = load_from_checkpoint(cfg)
    worker_index = os.environ.get('_worker_index_', None)
    store_path = os.environ.get('_store_path_', None)
    traj_num = os.environ.get('_traj_num_', None)
    resume_collection = os.environ.get('_resume_', False)
    worker_index = int(worker_index)
    traj_num = int(traj_num)
    resume_collection = True if resume_collection == 'True' else False
    max_traj_num = traj_num
    cfg.num_envs = 1
    save_interval = 100000
    cur_trans_num = 0
    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)
    env = make_env_func(AttrDict({'worker_index': worker_index, 'vector_index': 0}))
    # env.seed(0)
    is_multiagent = is_multiagent_env(env)
    if not is_multiagent:
        env = MultiAgentWrapper(env)
    task_name = env.level_name
    save_path = os.path.join(store_path, task_name)
    os.makedirs(save_path, exist_ok=True)
    avg_random_score = None
    if resume_collection:
        with open(os.path.join(save_path, '0_0_readme.json'), mode='r') as f:
            content = json.load(f)
        total_eps = content.get('Trajectory_num')
        total_transitions = content.get('Transition_num')
        total_ep_reward = content.get('Total_episode_return')
        avg_random_score = content.get('Average_random_return')
        min_ep_reward = content.get('Min_ep_reward')
        max_ep_reward = content.get('Max_ep_reward')
        min_ep_length = content.get('Min_ep_length')
        max_ep_length = content.get('Max_ep_length')
        save_idx = content.get('Save_idx', None)
        log.info('resume collection:')
        log.info(content)
        if save_idx is None:
            save_idx = 1000
        else:
            save_idx += 1
    else:
        max_ep_reward = -1e9
        min_ep_reward = 1e9
        max_ep_length = -1e9
        min_ep_length = 1e9
        save_idx = 0
        total_transitions = 0
        total_eps = 0
        total_ep_reward = 0  
    assert worker_index is not None and store_path is not None and traj_num is not None and save_idx is not None
    if hasattr(env.unwrapped, 'reset_on_init'):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False
    action_mapper = None
    if hasattr(env, 'action_set'):
        action_mapper = np.array(env.action_set, dtype=np.uint8)
    
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    index = os.environ.get('CUDA_VISIBLE_DEVICES', -1)
    device = torch.device('cpu' if cfg.device == 'cpu' or index == '-1' else 'cuda')
    print('device: ', device)
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])

    obsBuffer = []
    actionsBuffer = []
    rewardBuffer = []
    terminalBuffer = []
    traj_rews = []

    def max_trajs_reached(total_eps):
        return max_traj_num is not None and total_eps >= max_traj_num

    obs = env.reset()
    ep_steps = 0
    rnn_states = torch.zeros([env.num_agents, get_hidden_size(cfg)], dtype=torch.float32, device=device)

    data2save = reset_data()
    pid = os.getpid()
    with torch.no_grad():
        if avg_random_score is None:
            random_score = 0
            random_idx = 0
            while random_idx < 500:
                action = random.randint(0, len(action_mapper) - 1)
                obs, rew, done, infos = env.step([action])
                random_score += infos[0]['raw_step_rew']
                for agent_i, done_flag in enumerate(done):
                    if done_flag:
                        random_idx += 1
            avg_random_score = random_score / 500
            obs = env.reset()
        while not max_trajs_reached(total_eps):
            obs_torch = AttrDict(transform_dict_observations(obs))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(device).float()

            policy_outputs = actor_critic(obs_torch, rnn_states, with_action_distribution=True)

            # sample actions from the distribution by default
            actions = policy_outputs.actions

            action_distribution = policy_outputs.action_distribution
            if isinstance(action_distribution, ContinuousActionDistribution):
                if not cfg.continuous_actions_sample:  # TODO: add similar option for discrete actions
                    actions = action_distribution.means

            actions = actions.cpu().numpy()

            rnn_states = policy_outputs.rnn_states
            obsBuffer.append(obs[0]['obs'])
            obs, rew, done, infos = env.step(actions)
            rewardBuffer.append(infos[0]['raw_step_rew'])
            terminalBuffer.append(done[0])
            if action_mapper is None:
                actionsBuffer.append(actions[0])
            else:
                actionsBuffer.append(action_mapper[actions[0]])

            # print(obsBuffer)
            # print(actionsBuffer)
            # print(rewardBuffer)
            # print(terminalBuffer)
            # break
            ep_steps += 1

            for agent_i, done_flag in enumerate(done):
                if done_flag:
                    rnn_states[agent_i] = torch.zeros([get_hidden_size(cfg)], dtype=torch.float32, device=device)
                    status = append_data(data2save, obsBuffer, actionsBuffer, rewardBuffer, terminalBuffer)
                    if status:
                        total_eps += 1
                        traj_rews.append(infos[agent_i].get('true_reward', None))
                        cur_trans_num += ep_steps
                        log.info('pid: %d, Episode finished as %d trajs. true_reward: %.3f, cur trans: %d, save interval: %d', pid, total_eps, traj_rews[-1], cur_trans_num, save_interval)
                        min_ep_reward = min(min_ep_reward, traj_rews[-1])
                        max_ep_reward = max(max_ep_reward, traj_rews[-1])
                        min_ep_length = min(min_ep_length, ep_steps)
                        max_ep_length = max(max_ep_length, ep_steps)
                        
                        if cur_trans_num >= save_interval or total_eps >= max_traj_num:
                            total_transitions += cur_trans_num
                            log.info(f'pid: {pid} saving data')
                            dataset2save = h5py.File(os.path.join(save_path, str(save_idx) + '.hdf5'), 'w')
                            np.save(os.path.join(save_path, 'episode_reward_' + str(save_idx)), traj_rews)
                            total_ep_reward += np.sum(traj_rews)
                            npify(data2save)
                            for k in data2save:
                                dataset2save.create_dataset(k, data=data2save[k], compression='gzip')
                            res = {
                                'Trajectory_num': int(total_eps),
                                'Transition_num': int(total_transitions),
                                'Total_episode_return': round(total_ep_reward, 2),
                                'Average_episode_return': round(total_ep_reward / total_eps, 2),
                                'Average_random_return': round(avg_random_score, 2),
                                'Average_episode_trans': round(total_transitions / total_eps, 2),
                                'Min_ep_reward': round(min_ep_reward, 2),
                                'Max_ep_reward': round(max_ep_reward, 2),
                                'Min_ep_length': int(min_ep_length),
                                'Max_ep_length': int(max_ep_length),
                                'Save_idx': int(save_idx),                                
                            }
                            res_json = json.dumps(res)
                            with open(os.path.join(save_path, '0_0_readme.json'), 'w') as file:
                                file.write(res_json)
                            cur_trans_num = 0
                            traj_rews.clear()
                            save_idx += 1
                            del data2save
                            data2save = reset_data()
                    ep_steps = 0
                    obsBuffer.clear()
                    actionsBuffer.clear()
                    rewardBuffer.clear()
                    terminalBuffer.clear()

    env.close()
    return ExperimentStatus.SUCCESS


def main():
    """Script entry point."""
    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
