import argparse
import json
import os
import sys
import inspect

import torch.nn
import numpy as np

from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.qpg.sac import SAC
from rlpyt.algos.qpg.sacfd import SACfD
from rlpyt.agents.qpg.sac_agent import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.buffer import torchify_buffer, numpify_buffer
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.logging import logger
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.launching.variant import load_variant, update_config
from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.utils.launching.affinity import encode_affinity, quick_affinity_code
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel
from rlpyt.replays.non_sequence.uniform import (UniformReplayBuffer,
                                                AsyncUniformReplayBuffer)
from rlpyt.replays.non_sequence.time_limit import (TlUniformReplayBuffer,
                                                   AsyncTlUniformReplayBuffer)
import pickle
import psutil
# import gym_flySim


from utils.helper_functions import get_relevant_kwargs

# TODO: Importing SamplesToBufferTl here is done to avoid problems when loading expert demos. This can be avoided by switching to namedarraytupleschema in the creation of thr replayBuffer

SamplesToBuffer = namedarraytuple("SamplesToBuffer",
                                  ["observation", "action", "reward", "done"])
SamplesToBufferTl = namedarraytuple("SamplesToBufferTl",
                                    SamplesToBuffer._fields + ("timeout",))

ALGOS = {
    'SAC': SAC,
    'SACfD': SACfD
}
AGENTS = {
    'SacAgent': SacAgent
}
SAMPLERS = {
    'CpuSampler': CpuSampler,
    'SerialSampler': SerialSampler
}
RUNNERS = {
    'MinibatchRlEval': MinibatchRlEval
}


def build_and_train(config_path=None, slot_affinity_code=None, log_dir=None, run_ID=None):
    # First load the default configuration
    root_conf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './configurations/'))
    with open(os.path.join(root_conf_path, 'ops/single_conf.json'), 'r') as f:
        config = json.loads(f.read())
    # Overlay changes in alternative configuration file.
    # this doesn't have to be full, unset variables will stay with default values
    if config_path is not None:
        with open(config_path, 'r') as f:
            variant = json.loads(f.read())
            config = update_config(config, variant)

    # If given, get the affinity from code (usually assigned by batch mode)
    if slot_affinity_code is not None:
        affinity = affinity_from_code(slot_affinity_code)

    # If not given, use all available cpus up-to batch_B (maximal parallel environments)
    else:
        pp = psutil.Process()
        cpus = pp.cpu_affinity()
        cpus = cpus[:config['sampler']['batch_B']]
        affinity = dict(cuda_idx=None, workers_cpus=list([cpu] for cpu in cpus))

    # If a logging directory is provided and it has a viable configuration file (usually written by the batch mode)
    # Load it and make changes to our configuration (overriding previous values, unset changes are taken from default)
    if log_dir is not None:
        variant = load_variant(log_dir)
        config = update_config(config, variant)

    if config['sampler']['eval_env_kwargs'] == "env_kwargs":
        config['sampler']['eval_env_kwargs'] = config['sampler']['env_kwargs']

    agent_func = AGENTS[config['general']['agent']]
    r_kwargs = get_relevant_kwargs(agent_func, config['agent'])
    agent = agent_func(**r_kwargs)

    algo_func = ALGOS[config['general']['algo']]
    r_kwargs = get_relevant_kwargs(algo_func, config['algo'])
    algo = algo_func(**r_kwargs)

    sampler_func = SAMPLERS[config['general']['sampler']]
    r_kwargs = get_relevant_kwargs(sampler_func, config['sampler'])
    sampler = sampler_func(EnvCls=gym_make, **r_kwargs)

    runner_func = RUNNERS[config['general']['runner']]
    r_kwargs = get_relevant_kwargs(runner_func, config['runner'])
    runner = runner_func(algo=algo, agent=agent, sampler=sampler, affinity=affinity, **r_kwargs)

    if run_ID is not None:
        config['logger']['run_ID'] = run_ID
    if log_dir is not None:
        config['logger']['log_dir'] = log_dir
        config['logger']['override_prefix'] = True
    with logger_context(**config['logger']):
        exp_dir = logger.get_snapshot_dir()
        conf_filename = os.path.join(exp_dir, 'single_conf.json')
        with open(conf_filename, 'w') as f:
            f.writelines(json.dumps(config, indent=4))
        runner.train()


def get_saved_session_path(configuration_dict):
    log_dir = configuration_dict['logger']['log_dir']
    run_ID = configuration_dict['logger']['run_ID']
    log_dir = os.path.join(log_dir, f"run_{run_ID}")
    exp_dir = os.path.abspath(log_dir)
    saved_session_path = os.path.join(exp_dir, 'params.pkl')
    return saved_session_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='optional inputs for a single run')
    parser.add_argument('-c', '--config_path', type=str, default=None,
                        help='path for configuration file')
    parser.add_argument('-a', '--slot_affinity_code', type=str, default=None,
                        help='slot_affinity_code')
    parser.add_argument('-i', '--run_ID', type=int, default=None,
                        help='run_ID (int)')
    parser.add_argument('-d', '--log_dir', type=str, default=None,
                        help='logging directory (if this contains a configuration file named \'variant.json\',it will be used)')
    args = parser.parse_args()

    build_and_train(**vars(args))

    # default_configuration = {
    #     'general':
    #         {
    #             # 'sampler_type': 'SerialSampler',  # CpuSampler
    #             'sampler_type': 'CpuSampler',  # CpuSampler
    #             'algo':'SACfD'
    #         },
    #     'agent':
    #         {
    #             'model_kwargs':
    #                 {
    #                     'hidden_sizes': [64, 64, 32, 32]
    #                 },  # Pi model.
    #             'q_model_kwargs':
    #                 {
    #                     'hidden_sizes': [64, 64]
    #                 },
    #             'v_model_kwargs':
    #                 {
    #                     'hidden_sizes': [32, 32]
    #                 },
    #         },
    #     'algo':
    #         {
    #             'replay_size': 10000,
    #             'replay_ratio': 4,
    #             'batch_size': 256,
    #             'min_steps_learn': 256,
    #             'demonstrations_path': '/home/sagiv/Documents/HUJI/Tsevi/RL/rlpyt/data/fly_demo.pkl',
    #             'expert_ratio': 0.4,
    #             'expert_discount':0.6
    #         },
    #     'sampler':
    #         {
    #             'env_kwargs':
    #                 {
    #                     'id': 'gym_flySim:flySim-v0',
    #                     'config_path': None
    #                 },
    #             'eval_env_kwargs':
    #                 {
    #                     'id': 'gym_flySim:flySim-v0',
    #                     'config_path': None
    #                 },
    #             'max_decorrelation_steps': 0,  # Random sampling an action to bootstrap
    #             'eval_max_steps': 880,
    #             'eval_max_trajectories': 4,
    #
    #             'batch_T': 128,  # Environment steps per worker in batch
    #             'batch_B': 8,  # Total environments and agents
    #             'eval_n_envs': 4,
    #         },
    #     'runner':
    #         {
    #             'n_steps': 100000,  # Total environment steps
    #             'log_interval_steps': 1000,
    #         },
    #     'logger':
    #         {
    #             'log_dir': '/home/sagiv/Documents/HUJI/Tsevi/RL/rlpyt/data/flySim/',
    #             'run_ID': 111,
    #             'name': 'fly_SAC',
    #             'snapshot_mode': 'last',
    #             'use_summary_writer': True
    #         }
    # }
