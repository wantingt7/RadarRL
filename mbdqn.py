import os
import argparse
import numpy as np
import torch
import gym
# from Radar.SingleAgent.medqn_utils import DataBuffer
from tianshou.utils.net.common import Net
from tianshou.utils import BasicLogger
from tianshou.policy import DQNPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.trainer.utils import test_episode
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import RadarEnv


# Parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_real', type = str, default = 'RadarEnv-v0')
    parser.add_argument('--env_model', type = str, default = 'RadarEnvModel-v0')
    parser.add_argument('--jammer_policy_type', type = int, default = 3)
    parser.add_argument('--seed', type = int, default = 1626)
    parser.add_argument('--logdir', type=str, default='log_policy1/mbdqn_1626')
    parser.add_argument('--logname_fic', type=str, default='mbdqn_fic')
    parser.add_argument('--logname_temp', type=str, default='mbdqn_temp')
    parser.add_argument('--logname_real', type=str, default='mbdqn_real')
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
    parser.add_argument('--policy_hidden_sizes', type=int,
                    nargs='*', default=[128, 128, 128, 128])
    parser.add_argument('--policy_learning_rate', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n_step', type=int, default=3)
    parser.add_argument('--target_update_freq', type=int, default=320)
    parser.add_argument('--eps_train', type=float, default=0.1)
    parser.add_argument('--eps_test', type=float, default=0.05)
    parser.add_argument('--buffer_size_fic', type=int, default=20000)
    parser.add_argument('--policy_batch_size', type=int, default=64)
    parser.add_argument('--policy_training_num', type=int, default=10)
    parser.add_argument('--step_per_epoch', type=int, default=1000)
    parser.add_argument('--step_per_collect', type=int, default=100)
    parser.add_argument('--episode_per_test', type=int, default=100)
    parser.add_argument('--update_per_step', type=float, default=0.1)
    parser.add_argument('--model_type', type=int, default=2)

    args = parser.parse_args()
    return args


def main():
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
    policy_net = Net(args.state_shape, args.action_shape,
            hidden_sizes=args.policy_hidden_sizes, device=args.device,
            # dueling=(Q_param, V_param),
            ).to(args.device)
    optim = torch.optim.Adam(policy_net.parameters(), lr=args.policy_learning_rate)
    policy = DQNPolicy(
        policy_net, optim, args.gamma, args.n_step,
        target_update_freq=args.target_update_freq, 
        is_double=False)

    # Initialize collectors
    buffer_real = VectorReplayBuffer(args.buffer_size_real, buffer_num=1)
    train_collector_real = Collector(policy, env, buffer_real, exploration_noise=True)
    test_collector_real = Collector(policy, env, exploration_noise=True)
    # train collector fic will be initialized in the policy updating process
    test_collector_fic = Collector(policy, env_model, exploration_noise=True)

    # ME-DQN
    # Record the reward
    best_reward, best_reward_std = 0, 0

    ### utils
    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(logpath_real, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold

    def train_fn(epoch, env_step):
        if env_step <= 10000:
            policy.set_eps(args.eps_train)
        elif env_step <= 50000:
            eps = args.eps_train - (env_step - 10000) / \
                40000 * (0.9 * args.eps_train)
            policy.set_eps(eps)
        else:
            policy.set_eps(0.1 * args.eps_train)

    def test_fn(epoch, env_step):
        policy.set_eps(args.eps_test)

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        pass

    ### Outer loop
    env_step = 0
    
    # Test the policy in the real environment
    test_result = test_episode(policy, test_collector_real, test_fn, 0, args.episode_per_test, logger_real, env_step)
    rew, rew_std = test_result["rew"], test_result["rew_std"]
    if best_reward < rew:
        best_reward, best_reward_std = rew, rew_std
    print(f"Initialize: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
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

        # Train the policy with fic samples
        # Copy all the real training samples and keep adding the newest fictitious samples with new fn
        buffer_fic = VectorReplayBuffer(args.buffer_size_fic, buffer_num=1)
        train_collector_fic = Collector(policy, env_model, buffer_fic, exploration_noise=True)
        train_collector_fic.deepcopy(train_collector_real)

        _, test_result = offpolicy_trainer(
            policy, train_collector_fic, test_collector_fic, args.inner_epoch_num,         # inner loop implied here
            args.step_per_epoch, args.step_per_collect, args.episode_per_test,
            args.policy_batch_size, update_per_step=args.update_per_step, train_fn=train_fn,
            test_fn=test_fn, stop_fn=stop_fn, save_fn=save_fn, 
            save_checkpoint_fn=save_checkpoint_fn, logger=logger_temp)
        logger_fic.log_test_data(test_result, env_step)
        # assert stop_fn(result['best_reward'])

        # Test the policy in the real environment
        test_result = test_episode(policy, test_collector_real, test_fn, 0, args.episode_per_test, logger_real, env_step)
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        if best_reward < rew:
            best_reward, best_reward_std = rew, rew_std
        print(f"Epoch ##{epoch1}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
            f"ard: {best_reward:.6f} ± {best_reward_std:.6f}")



if __name__ == '__main__':
    main()
