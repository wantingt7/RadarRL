import os
import gym
import torch
import pprint
import argparse
import numpy as np
import RadarEnv
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import BasicLogger
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.net.discrete import Actor, Critic
from tianshou.policy import A2CPolicy, ImitationPolicy
from tianshou.trainer import onpolicy_trainer, offpolicy_trainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='RadarEnv-v0')
    parser.add_argument('--jammer_policy_type', type = int, default = 3)
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--il-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--il-step-per-epoch', type=int, default=1000)
    parser.add_argument('--episode-per-collect', type=int, default=16)
    parser.add_argument('--step-per-collect', type=int, default=100)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--hidden-sizes', type=int,
                        nargs='*', default=[64, 64])
    parser.add_argument('--imitation-hidden-sizes', type=int,
                        nargs='*', default=[128])
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=100)
    parser.add_argument('--logdir', type=str, default='log_temp/test8/policy2/a2c')
    parser.add_argument('--logname-basic', type=str, default='a2c_basic')
    parser.add_argument('--logname-il', type=str, default='a2c_il')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    # a2c special
    parser.add_argument('--vf-coef', type=float, default=0.5)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--max-grad-norm', type=float, default=None)
    parser.add_argument('--gae-lambda', type=float, default=1.)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    args = parser.parse_known_args()[0]
    return args


def test_a2c_with_il(args=get_args()):
    torch.set_num_threads(1)  # for poor CPU
    env = gym.make(args.task)
    env.set_jammer_type(args.jammer_policy_type)
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(args.task)
    train_envs_list = []
    for _ in range(args.training_num):
        e = gym.make(args.task)
        e.set_jammer_type(args.jammer_policy_type)
        train_envs_list.append(lambda:e)
    train_envs = DummyVectorEnv(train_envs_list)
    # test_envs = gym.make(args.task)
    test_envs_list = []
    for _ in range(args.test_num):
        e = gym.make(args.task)
        e.set_jammer_type(args.jammer_policy_type)
        test_envs_list.append(lambda:e)
    test_envs = DummyVectorEnv(test_envs_list)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)
    # model
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
              device=args.device)
    actor = Actor(net, args.action_shape, device=args.device).to(args.device)
    critic = Critic(net, device=args.device).to(args.device)
    optim = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()), lr=args.lr)
    dist = torch.distributions.Categorical
    policy = A2CPolicy(
        actor, critic, optim, dist,
        discount_factor=args.gamma, gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef, ent_coef=args.ent_coef,
        max_grad_norm=args.max_grad_norm, reward_normalization=args.rew_norm,
        action_space=env.action_space)
    # collector
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)))
    test_collector = Collector(policy, test_envs)
    # log
    log_path_basic = os.path.join(args.logdir, args.logname_basic)
    logger_basic = BasicLogger(SummaryWriter(log_path_basic))

    def save_fn_basic(policy):
        torch.save(policy.state_dict(), os.path.join(log_path_basic, 'policy.pth'))

    def stop_fn(mean_rewards):
        return mean_rewards >= env.spec.reward_threshold

    # trainer
    result = onpolicy_trainer(
        policy, train_collector, test_collector, args.epoch,
        args.step_per_epoch, args.repeat_per_collect, args.test_num, args.batch_size,
        episode_per_collect=args.episode_per_collect, stop_fn=stop_fn, save_fn=save_fn_basic,
        logger=logger_basic)
    # assert stop_fn(result['best_reward'])

    if __name__ == '__main__':
        pprint.pprint(result)
        # Let's watch its performance!
        env = gym.make(args.task)
        env.set_jammer_type(args.jammer_policy_type)
        policy.eval()
        collector = Collector(policy, env)
        result = collector.collect(n_episode=1, render=args.render)
        rews, lens = result["rews"], result["lens"]
        print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


    # # log
    # log_path_il = os.path.join(args.logdir, args.logname_il)
    # logger_il = BasicLogger(SummaryWriter(log_path_il))

    # def save_fn_il(policy):
    #     torch.save(policy.state_dict(), os.path.join(log_path_il, 'policy.pth'))

    # policy.eval()
    # # here we define an imitation collector with a trivial policy
    # if args.task == 'CartPole-v0':
    #     env.spec.reward_threshold = 190  # lower the goal
    # net = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
    #           device=args.device)
    # net = Actor(net, args.action_shape, device=args.device).to(args.device)
    # optim = torch.optim.Adam(net.parameters(), lr=args.il_lr)
    # il_policy = ImitationPolicy(net, optim, action_space=env.action_space)
    # il_test_collector = Collector(
    #     il_policy,
    #     DummyVectorEnv([lambda: gym.make(args.task) for _ in range(args.test_num)])
    # )
    # train_collector.reset()
    # result = offpolicy_trainer(
    #     il_policy, train_collector, il_test_collector, args.epoch,
    #     args.il_step_per_epoch, args.step_per_collect, args.test_num,
    #     args.batch_size, stop_fn=stop_fn, save_fn=save_fn_il, logger=logger_il)
    # assert stop_fn(result['best_reward'])

    # if __name__ == '__main__':
    #     pprint.pprint(result)
    #     # Let's watch its performance!
    #     env = gym.make(args.task)
    #     il_policy.eval()
    #     collector = Collector(il_policy, env)
    #     result = collector.collect(n_episode=1, render=args.render)
    #     rews, lens = result["rews"], result["lens"]
    #     print(f"Final reward: {rews.mean()}, length: {lens.mean()}")


if __name__ == '__main__':
    test_a2c_with_il()
