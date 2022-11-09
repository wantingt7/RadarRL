import os
import argparse
import numpy as np
import torch
import gym
# from Radar.SingleAgent.medqn_utils import DataBuffer
from tianshou.utils.net.common import Net
from tianshou.utils import BasicLogger
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.trainer.utils import test_episode
from tianshou.data import Collector, VectorReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import RadarEnv


# Parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_real', type = str, default = 'RadarEnv-v0')
    parser.add_argument('--env_model', type = str, default = 'RadarEnv-v2')
    parser.add_argument('--seed', type = int, default = 1626)
    parser.add_argument('--logdir', type=str, default='log_temp/test5/mbppo')
    parser.add_argument('--logname_fic', type=str, default='mbppo_fic')
    parser.add_argument('--logname_temp', type=str, default='mbppo_temp')
    parser.add_argument('--logname_real', type=str, default='mbppo_real')
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
    parser.add_argument('--policy_hidden_sizes', type=int, nargs='*', default=[64, 64])
    parser.add_argument('--policy_learning_rate', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--training_num', type=int, default=1)
    parser.add_argument('--test_num', type=int, default=1)
    parser.add_argument('--n_step', type=int, default=3)
    parser.add_argument('--buffer_size_fic', type=int, default=20000)
    parser.add_argument('--policy_batch_size', type=int, default=64)
    parser.add_argument('--step_per_epoch', type=int, default=100)
    parser.add_argument('--step_per_collect', type=int, default=50)
    parser.add_argument('--repeat-per-collect', type=int, default=10)
    parser.add_argument('--episode_per_test', type=int, default=100)
    parser.add_argument('--update_per_step', type=float, default=0.1)
    # general parameters
    parser.add_argument('--model_type', type=int, default=2)
    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=0)
    parser.add_argument('--norm-adv', type=int, default=0)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=0)

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
    logpath_fic = os.path.join(args.logdir, args.logname_fic)
    logger_fic = BasicLogger(SummaryWriter(logpath_fic))
    logpath_temp = os.path.join(args.logdir, args.logname_temp)
    logger_temp = BasicLogger(SummaryWriter(logpath_temp))
    logpath_real = os.path.join(args.logdir, args.logname_real)
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
    policy_net = Net(args.state_shape, hidden_sizes=args.policy_hidden_sizes,
              device=args.device)
    actor = Actor(policy_net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(policy_net, device=args.device).to(args.device)
    # orthogonal initialization
    for m in list(actor.modules()) + list(critic.modules()):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.policy_learning_rate)
    dist = torch.distributions.Categorical
    policy = PPOPolicy(
        actor, critic, optim, dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        gae_lambda=args.gae_lambda,
        reward_normalization=args.rew_norm,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        action_space=env.action_space,
        deterministic_eval=True,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv)

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
            return mean_rewards >= env_model.spec.reward_threshold
        
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