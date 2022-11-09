import os
import gym
import torch
import pprint
import argparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import BasicLogger
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.policy import A2CPolicy, ImitationPolicy
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer
from tianshou.trainer.utils import test_episode

import RadarEnv


# Parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_real', type = str, default = 'RadarEnv-v0')
    parser.add_argument('--env_model', type = str, default = 'RadarEnvModel-v0')
    parser.add_argument('--jammer_policy_type', type = int, default = 3)
    parser.add_argument('--seed', type = int, default = 1626)
    parser.add_argument('--logdir', type=str, default='log/mba2c_type3')
    parser.add_argument('--logname_fic', type=str, default='mba2c_fic')
    parser.add_argument('--logname_temp', type=str, default='mba2c_temp')
    parser.add_argument('--logname_real', type=str, default='mba2c_real')
    parser.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    # parameters for model
    parser.add_argument('--model_num', type=int, default=1)
    parser.add_argument('--outer_epoch_num', type=int, default=50)
    parser.add_argument('--buffer_size_real', type=int, default=20000)
    parser.add_argument('--model_learning_rate', type=float, default=1e-3)
    parser.add_argument('--model_batch_size', type=int, default=100)
    parser.add_argument('--model_batch_num', type=int, default=10)
    parser.add_argument('--model_epoch_num', type=int, default=10)
    # parameters for policy
    parser.add_argument('--inner_epoch_num', type=int, default=20)
    parser.add_argument('--buffer_size_fic', type=int, default=20000)
    parser.add_argument('--policy_learning_rate', type=float, default=1e-3)
    parser.add_argument('--il-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--training_num', type=int, default=1)
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--n_step', type=int, default=3)
    parser.add_argument('--policy_batch_size', type=int, default=64)
    parser.add_argument('--policy_hidden_sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--imitation-hidden-sizes', type=int,
                        nargs='*', default=[128])
    parser.add_argument('--step_per_epoch', type=int, default=1000)
    parser.add_argument('--step_per_collect', type=int, default=100)
    parser.add_argument('--episode_per_test', type=int, default=100)
    parser.add_argument('--il-step-per-epoch', type=int, default=100)
    parser.add_argument('--episode-per-collect', type=int, default=16)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--update_per_step', type=float, default=0.1)
    parser.add_argument('--reward-threshold', type=float, default=None)
    # general parameters
    parser.add_argument('--model_type', type=int, default=2)
    # a2c special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=1.)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    args = parser.parse_args()
    return args


def main():
    torch.set_num_threads(1)  # for poor CPU
    args = get_args()
    # Set seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Log experiment data
    logpath_fic = os.path.join(args.logdir, args.logname_fic)
    logger_fic = BasicLogger(SummaryWriter(logpath_fic))
    logpath_temp = os.path.join(args.logdir, args.logname_temp)
    logger_temp = BasicLogger(SummaryWriter(logpath_temp))
    logpath_real = os.path.join(args.logdir, args.logname_real)
    logger_real = BasicLogger(SummaryWriter(logpath_real))

    # Initialize environment
    env = gym.make(args.env_real)
    env.set_jammer_type(args.jammer_policy_type)

    # Initialize model
    env_model = gym.make(args.env_model)
    env_model.set_jammer_type(args.jammer_policy_type)
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
    policy_net = Net(args.state_shape, hidden_sizes=args.policy_hidden_sizes,
              device=args.device)
    actor = Actor(policy_net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(policy_net, device=args.device).to(args.device)
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.policy_learning_rate)
    dist = torch.distributions.Categorical
    policy = A2CPolicy(
        actor, critic, optim, dist,
        discount_factor=args.gamma, gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef, ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm, reward_normalization=args.rew_norm,
        action_space=env.action_space)

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

    # Test the policy in the real environment
    test_result = test_episode(policy, test_collector_real, None, 0, args.episode_per_test, logger_real, env_step)
    rew, rew_std = test_result["rew"], test_result["rew_std"]
    if best_reward < rew:
        best_reward, best_reward_std = rew, rew_std
    print(f"Epoch ##{epoch1}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
        f"ard: {best_reward:.6f} ± {best_reward_std:.6f}")

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
            return mean_rewards >= env.spec.reward_threshold
        
        # Copy all the real training samples and keep adding the newest fictitious samples with new fn
        buffer_fic = VectorReplayBuffer(args.buffer_size_fic, buffer_num=1)
        train_collector_fic = Collector(policy, env_model, buffer_fic, exploration_noise=True)
        train_collector_fic.deepcopy(train_collector_real)

        _, test_result = onpolicy_trainer(
            policy, train_collector_fic, test_collector_fic, args.inner_epoch_num,
            args.step_per_epoch, args.repeat_per_collect, args.test_num, args.policy_batch_size,
            episode_per_collect=args.episode_per_collect, stop_fn=stop_fn, save_fn=save_fn, 
            logger=logger_temp)
        logger_fic.log_test_data(test_result, env_step)
        # assert stop_fn(result['best_reward'])

        '''
        policy.eval()
        # here we define an imitation collector with a trivial policy
        if args.task == 'CartPole-v0':
            env.spec.reward_threshold = 190  # lower the goal
        net = Net(args.state_shape, hidden_sizes=args.policy_hidden_sizes,
                device=args.device)
        net = Actor(net, args.action_shape, device=args.device).to(args.device)
        optim = torch.optim.Adam(net.parameters(), lr=args.il_lr)
        il_policy = ImitationPolicy(net, optim, action_space=env.action_space)
        il_test_collector = Collector(il_policy, env_model, exploration_noise=True)
        train_collector_fic.reset()
        _, test_result = offpolicy_trainer(
            il_policy, train_collector_fic, il_test_collector, args.inner_epoch_num,
            args.il_step_per_epoch, args.step_per_collect, args.test_num,
            args.policy_batch_size, stop_fn=stop_fn, save_fn=save_fn, 
            resume_from_log = True, logger=logger_temp)
        logger_fic.log_test_data(test_result, env_step)
        # assert stop_fn(result['best_reward'])
        '''

        # Test the policy in the real environment
        test_result = test_episode(policy, test_collector_real, None, 0, args.episode_per_test, logger_real, env_step)
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        if best_reward < rew:
            best_reward, best_reward_std = rew, rew_std
        print(f"Epoch ##{epoch1}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
            f"ard: {best_reward:.6f} ± {best_reward_std:.6f}")



if __name__ == '__main__':
    main()