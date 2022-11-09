import argparse
import os

import gym
import numpy as np

import torch
from torch import nn
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import TRPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import BasicLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import Actor, Critic 
from tianshou.trainer.utils import test_episode

import RadarEnv


# Parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_real', type = str, default = 'RadarEnv-v0')
    parser.add_argument('--env_model', type = str, default = 'RadarEnv-v2')
    parser.add_argument('--seed', type = int, default = 1626)
    parser.add_argument('--logdir', type=str, default='log_me_cc/metrpo_cc')
    parser.add_argument('--logname_fic', type=str, default='metrpo_fic')
    parser.add_argument('--logname_temp', type=str, default='metrpo_temp')
    parser.add_argument('--logname_real', type=str, default='metrpo_real')
    parser.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    # parameters for model
    parser.add_argument('--model_num', type=int, default=1)
    parser.add_argument('--outer_epoch_num', type=int, default=80)
    parser.add_argument('--buffer_size_real', type=int, default=20000)
    parser.add_argument('--model_learning_rate', type=float, default=1e-3)
    parser.add_argument('--model_batch_size', type=int, default=100)
    parser.add_argument('--model_batch_num', type=int, default=10)
    parser.add_argument('--model_epoch_num', type=int, default=10)
    # parameters for policy
    parser.add_argument('--inner_epoch_num', type=int, default=50)
    parser.add_argument('--buffer_size_fic', type=int, default=20000)
    parser.add_argument('--policy_learning_rate', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--training_num', type=int, default=1)
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--n_step', type=int, default=3)
    parser.add_argument('--policy_batch_size', type=int, default=64)
    parser.add_argument('--policy_hidden_sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--step_per_epoch', type=int, default=20)
    parser.add_argument('--step_per_collect', type=int, default=10)
    parser.add_argument('--repeat-per-collect', type=int, default=2)  # theoretically it should be 1
    parser.add_argument('--episode_per_test', type=int, default=100)
    parser.add_argument('--update_per_step', type=float, default=0.1)
    parser.add_argument('--reward-threshold', type=float, default=None)
    # general parameters
    parser.add_argument('--model_type', type=int, default=2)
    # trpo special
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--optim-critic-iters', type=int, default=5)
    parser.add_argument('--max-kl', type=float, default=0.005)
    parser.add_argument('--backtrack-coeff', type=float, default=0.8)
    parser.add_argument('--max-backtracks', type=int, default=10)
    args = parser.parse_args()
    return args


def train_models(model, samples_real):
    pass

# Use dqn policy in tianshou
def train_policy(samples_fic):
    pass

def test_policy_fic():
    pass

def test_policy_real():
    pass


def main():
    args = get_args()
    # Set seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Log experiment data
    logpath_fic = os.path.join(args.logdir, args.env_real, args.logname_fic)
    logger_fic = BasicLogger(SummaryWriter(logpath_fic))
    logpath_temp = os.path.join(args.logdir, args.env_real, args.logname_temp)
    logger_temp = BasicLogger(SummaryWriter(logpath_temp))
    logpath_real = os.path.join(args.logdir, args.env_real, args.logname_real)
    logger_real = BasicLogger(SummaryWriter(logpath_real))

    # Initialize environment
    env = gym.make(args.env_real)

    # Initialize model
    env_model = gym.make(args.env_model)
    # check model type
    if args.model_type == 2:
        env_model.set_type(args.model_type)
    env_model.initialize_network(args.model_num, args.model_learning_rate)


    # Initialize policy
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # args.max_action = env.action_space.high[0] 
    if args.reward_threshold is None:
        default_reward_threshold = {"CartPole-v0": 195}
        args.reward_threshold = default_reward_threshold.get(
            args.env_model, env_model.spec.reward_threshold
        )
    policy_net = Net(args.state_shape, hidden_sizes=args.policy_hidden_sizes, activation=nn.Tanh,
              device=args.device)
    actor = Actor(policy_net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(policy_net, device=args.device).to(args.device)
    # orthogonal initialization
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(critic.parameters(), lr=args.policy_learning_rate)
    dist = torch.distributions.Categorical
    policy = TRPOPolicy(
        actor, critic, optim, dist,
        discount_factor=args.gamma,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv,
        gae_lambda=args.gae_lambda,
        action_space=env.action_space,
        optim_critic_iters=args.optim_critic_iters,
        max_kl=args.max_kl,
        backtrack_coeff=args.backtrack_coeff,
        max_backtracks=args.max_backtracks)

    # Initialize collectors
    buffer_real = VectorReplayBuffer(args.buffer_size_real, buffer_num=1)
    train_collector_real = Collector(policy, env, buffer_real, exploration_noise=True)
    test_collector_real = Collector(policy, env, exploration_noise=True)
    buffer_fic = VectorReplayBuffer(args.buffer_size_fic, buffer_num=1)
    train_collector_fic = Collector(policy, env_model, buffer_fic, exploration_noise=True)
    test_collector_fic = Collector(policy, env_model, exploration_noise=True)

    # ME-PPO
    # Record the reward
    best_reward, best_reward_std = 0, 0

    ### Outer loop
    env_step = 0
    for epoch1 in range(args.outer_epoch_num):
        # Collect real samples
        train_collector_real.collect(n_step=args.model_batch_size * args.model_batch_num)
        env_step += args.model_batch_size * args.model_batch_num
        # Train model
        env_model.update_network(train_collector_real, args.model_batch_size, args.model_batch_num, args.model_epoch_num)


        ### Inner loop
        # Collect fic samples
        # train_collector_fic.collect(n_step=args.policy_batch_size * args.policy_training_num)

        ### Train ppo policy in the environment model
        def save_fn(policy):
            torch.save(policy.state_dict(), os.path.join(logpath_fic, 'policy.pth'))

        def stop_fn(mean_rewards):
            return mean_rewards >= args.reward_threshold
        
        # Copy all the real training samples and keep adding the newest fictitious samples with new fn
        buffer_fic = VectorReplayBuffer(args.buffer_size_fic, buffer_num=1)
        train_collector_fic = Collector(policy, env_model, buffer_fic, exploration_noise=True)
        train_collector_fic.deepcopy(train_collector_real)

        _, test_result = onpolicy_trainer(
            policy, train_collector_fic, test_collector_fic, args.inner_epoch_num,
            args.step_per_epoch, args.repeat_per_collect, args.test_num, args.policy_batch_size,
            step_per_collect=args.step_per_collect, stop_fn=stop_fn, save_fn=save_fn,
            resume_from_log = True, logger=logger_temp)
        logger_fic.log_test_data(test_result, env_step)
        # assert stop_fn(result['best_reward'])

        # Test the policy in the real environment
        test_result = test_episode(policy, test_collector_real, None, 0, args.episode_per_test, logger_real, env_step)
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        if best_reward < rew:
            best_reward, best_reward_std = rew, rew_std
        print(f"Epoch ##{epoch1}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
            f"ard: {best_reward:.6f} ± {best_reward_std:.6f}")



if __name__ == '__main__':
    main()