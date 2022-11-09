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
Introdution:
1. This model builds a network to learn the radar's next action.
2. The policy of radar always follows a uniform distribution. In other words, radar always randomly pick an action.
3. The main propose of this code is to test how many samples we should collect so that our network can predict the radar's action very well.
'''


# Parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_real', type = str, default = 'RadarEnv-v0')
    parser.add_argument('--env_model', type = str, default = 'RadarEnvModel-v0')
    parser.add_argument('--seed', type = int, default = 1626)
    parser.add_argument('--logdir', type=str, default='log_test/ficmodel_test_1')
    # parser.add_argument('--logname_fic', type=str, default='medqn_fic')
    # parser.add_argument('--logname_temp', type=str, default='medqn_temp')
    # parser.add_argument('--logname_real', type=str, default='medqn_real')
    parser.add_argument('--logname_test', type=str, default='ficnet_test')
    parser.add_argument('--device', type=str,
                    default='cuda' if torch.cuda.is_available() else 'cpu')
    # parameters for model
    parser.add_argument('--model_num', type=int, default=1)         # only train one fictitious model
    parser.add_argument('--outer_epoch_num', type=int, default=50)    # modified
    parser.add_argument('--buffer_size_real', type=int, default=20000)
    parser.add_argument('--model_learning_rate', type=float, default=1e-3)
    parser.add_argument('--model_batch_size', type=int, default=100)
    parser.add_argument('--model_batch_num', type=int, default=10)
    parser.add_argument('--model_epoch_num', type=int, default=10)
    # parameters for policy
    parser.add_argument('--inner_epoch_num', type=int, default=50)
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
    parser.add_argument('--step_per_epoch', type=int, default=20)
    parser.add_argument('--step_per_collect', type=int, default=10)
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
    logpath_test = os.path.join(args.logdir, args.logname_test)
    logger_test = BasicLogger(SummaryWriter(logpath_test))

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
    buffer_train = VectorReplayBuffer(args.buffer_size_real, buffer_num=1)
    train_collector = Collector(policy, env, buffer_train, exploration_noise=True)
    buffer_test = VectorReplayBuffer(args.buffer_size_real, buffer_num=1)
    test_collector = Collector(policy, env, buffer_test, exploration_noise=True)

    # Get the testing data
    test_collector.collect(n_step=args.test_data_size)

    ### Outer loop
    env_step = 0
    for epoch1 in range(args.outer_epoch_num):
        # Collect real samples
        train_collector.collect(n_step=args.model_batch_size * args.model_batch_num)
        env_step += args.model_batch_size * args.model_batch_num
        # Train model
        env_model.update_network(train_collector, args.model_batch_size, args.model_batch_num, args.model_epoch_num)

        # Test the model's prediction with the real outputs
        loss = env_model.test_accuracy(0, test_collector, args.test_data_size, args.test_loss_fn)
        print(f"Epoch ##{epoch1}: test_loss: {loss:.6f}")
        
        # Write the loss into the logger
        if env_step - logger_test.last_log_test_step >= logger_test.test_interval:
            logger_test.writer.add_scalar("test/loss", loss, global_step=env_step)
            logger_test.last_log_test_step = env_step


if __name__ == '__main__':
    main()
