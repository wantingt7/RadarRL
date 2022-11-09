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

'''
This code contains 2 parts:
1. Train the model with certain samples (following uniform distribution) until converge
2. Run dqn algorithm on the trained model, see the best rewards it can achieve when it converges
'''


# Parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_real', type = str, default = 'RadarEnv-v0')
    parser.add_argument('--env_model', type = str, default = 'RadarEnv-v2')
    parser.add_argument('--seed', type = int, default = 1626)
    parser.add_argument('--logdir', type=str, default='log1/ficmodel_dqn_2w')
    parser.add_argument('--logname_dqn_fic', type=str, default='dqn_fic')
    parser.add_argument('--logname_dqn_real_test', type=str, default='dqn_real_test')
    parser.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    # parameters for model
    parser.add_argument('--model_num', type=int, default=1)         # only train one fictitious model
    parser.add_argument('--model_epoch_num', type=int, default=50)    # epoch number when training model
    parser.add_argument('--buffer_size_model_train', type=int, default=20000)
    parser.add_argument('--buffer_size_model_test', type=int, default=20000)
    parser.add_argument('--model_learning_rate', type=float, default=1e-3)
    parser.add_argument('--model_batch_size', type=int, default=100)
    parser.add_argument('--model_batch_num', type=int, default=200)
    parser.add_argument('--model_inner_epoch_num', type=int, default=5)
    # parameters for policy
    parser.add_argument('--dqn_epoch_num', type=int, default=80)     # epoch number when training policy on model
    parser.add_argument('--dqn_epoch_between_real_test', type=float, default=5)     # frequency to test policy in real environment
    parser.add_argument('--policy_hidden_sizes', type=int,
                    nargs='*', default=[128, 128, 128, 128])
    parser.add_argument('--policy_learning_rate', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--n_step', type=int, default=3)
    parser.add_argument('--target_update_freq', type=int, default=320)
    parser.add_argument('--eps_train', type=float, default=0.1)
    parser.add_argument('--eps_test', type=float, default=0.05)
    parser.add_argument('--buffer_size_dqn_train', type=int, default=20000)
    parser.add_argument('--policy_batch_size', type=int, default=64)
    parser.add_argument('--policy_training_num', type=int, default=10)
    parser.add_argument('--step_per_epoch', type=int, default=1000)
    parser.add_argument('--step_per_collect', type=int, default=100)
    parser.add_argument('--episode_per_test', type=int, default=100)
    parser.add_argument('--update_per_step', type=float, default=0.1)
    parser.add_argument('--model_type', type=int, default=2)
    parser.add_argument('--test_data_size', type=int, default=1000)
    # model_type = 1, loss_fn = "MSELoss"
    # model_type = 2, loss_fn = "CrossEntropyLoss" / "KLDivLoss"
    parser.add_argument('--test_loss_fn', type=str, default="KLDivLoss")
    

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Set seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Log experiment data
    logpath_dqn_fic = os.path.join(args.logdir, args.logname_dqn_fic)       
    logger_dqn_fic = BasicLogger(SummaryWriter(logpath_dqn_fic))                     # contain both train & test loss with respect to fic env
    logpath_dqn_real_test = os.path.join(args.logdir, args.logname_dqn_real_test)
    logger_dqn_real_test = BasicLogger(SummaryWriter(logpath_dqn_real_test))

    # Initialize environment
    env = gym.make(args.env_real)

    # Initialize model
    env_model = gym.make(args.env_model)
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

    # Initialize buffer and collectors
    model_train_buffer = VectorReplayBuffer(args.buffer_size_model_train, buffer_num=1)
    model_train_collector = Collector(policy, env, model_train_buffer, exploration_noise=True)
    model_test_buffer = VectorReplayBuffer(args.buffer_size_model_test, buffer_num=1)
    model_test_collector = Collector(policy, env, model_test_buffer, exploration_noise=True)
    dqn_train_buffer = VectorReplayBuffer(args.buffer_size_dqn_train, buffer_num=1)
    dqn_train_collector = Collector(policy, env_model, dqn_train_buffer, exploration_noise=True)
    dqn_test_fic_collector = Collector(policy, env_model, exploration_noise=True)
    dqn_test_real_collector = Collector(policy, env, exploration_noise=True)

    # Get the training and testing data
    model_train_collector.collect(n_step=args.model_batch_size * args.model_batch_num)
    model_test_collector.collect(n_step=args.test_data_size)

    ### Train the model with uniform distributed samples
    for epoch1 in range(args.model_epoch_num):
        # Train model
        env_model.update_network(model_train_collector, args.model_batch_size, args.model_batch_num, args.model_inner_epoch_num)

        # Test the model's prediction with the real outputs
        loss = env_model.test_accuracy(0, model_test_collector, args.test_data_size, args.test_loss_fn)
        print(f"Model Epoch ##{epoch1}: test_loss: {loss:.6f}")


    ### Train dqn policy in the environment model
    def save_fn(policy):         
        torch.save(policy.state_dict(), os.path.join(logpath_dqn_fic, 'policy.pth'))

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

    # print the reward when step = 0
    env_step_fic = 0
    print('*', env_step_fic)
    test_result = test_episode(policy, dqn_test_real_collector, test_fn, 0, args.episode_per_test, logger_dqn_real_test, env_step_fic)
    best_reward, best_reward_std = test_result["rew"], test_result["rew_std"]

    for epoch2 in range(args.dqn_epoch_between_real_test, args.dqn_epoch_num, args.dqn_epoch_between_real_test):
        _, test_result = offpolicy_trainer(
                policy, dqn_train_collector, dqn_test_fic_collector, epoch2,
                args.step_per_epoch, args.step_per_collect, args.episode_per_test,
                args.policy_batch_size, update_per_step=args.update_per_step, train_fn=train_fn,
                test_fn=test_fn, stop_fn=stop_fn, save_fn=save_fn, 
                save_checkpoint_fn=save_checkpoint_fn,  resume_from_log = True, logger=logger_dqn_fic)      # set resume_from_log as true

        # Test the policy in the real environment
        _, env_step_fic, _ = logger_dqn_fic.restore_data()
        print('*', env_step_fic)
        test_result = test_episode(policy, dqn_test_real_collector, test_fn, 0, args.episode_per_test, logger_dqn_real_test, env_step_fic)
        rew, rew_std = test_result["rew"], test_result["rew_std"]
        if best_reward < rew:
            best_reward, best_reward_std = rew, rew_std
        print(f"DQN Epoch ##{epoch2}: test_reward: {rew:.6f} ± {rew_std:.6f}, best_rew"
            f"ard: {best_reward:.6f} ± {best_reward_std:.6f}")


if __name__ == '__main__':
    main()
