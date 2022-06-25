import argparse
import numpy as np
import random
import torch

from TD3.env import FractalEnv
from TD3.brain import TD3Agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=None,
                        help='The batch size')
    parser.add_argument('--ckp-path', type=str, default="logs/ckps",
                        help='The path for ckp saving')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use. Like cuda, cuda:0 or cpu')
    parser.add_argument('--gamma', type=float, default=None,
                        help='The discount factor')
    parser.add_argument('--lr-actor', type=float, default=None,
                        help='The learning rate for the actor')
    parser.add_argument('--lr-critic', type=float, default=None,
                        help='The learning rate for the critic')
    parser.add_argument('--memory-size', type=int, default=None,
                        help='The size of the replay buffer')
    parser.add_argument('--predictor-ckp', type=str, default=None,
                        help='The ckp path for the S-FID predictor')
    parser.add_argument('--random-steps', type=int, default=None,
                        help='The initial random action episodes')
    parser.add_argument('--seed', type=int, default=None, )
    args = parser.parse_args()

    seed = 42 if args.seed is None else args.seed  # lucky seed :)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(args.device)

    # (default) parameters
    num_steps = 5000000
    batch_size = 512 if args.batch_size is None else args.batch_size
    gamma = 0.9 if args.gamma is None else args.gamma
    lr_actor = 1e-4 if args.lr_actor is None else args.lr_actor
    lr_critic = 3e-4 if args.lr_critic is None else args.lr_critic
    memory_size = 500000 if args.memory_size is None else args.memory_size  # 100000
    initial_random_steps = 10000 if args.random_steps is None else args.random_steps
    assert args.predictor_ckp is not None

    # create the environment
    env = FractalEnv(obs_dim=(3, 224, 224),
                     train_root="datasets/rl/train",
                     action_range=[(0, 0.5), (0, 10000), (0, 60), (-90, 90)],
                     device=device,
                     predictor_ckp=args.predictor_ckp)

    # create the agent
    opts = {
        'batch_size': batch_size,
        'ckp_path': args.ckp_path,
        'device': device,
        'encoder_path': "msm/backbone/mobilenet_v2-b0353104.pth",
        'gamma': gamma,
        'initial_random_steps': initial_random_steps,
        'lr_actor': lr_actor,
        'lr_critic': lr_critic,
        'memory_size': memory_size,
    }
    exploration_noise = {
        'min_sigma': 0.1,
        'max_sigma': 0.1,
        'decay_period': 100000,
    }
    agent = TD3Agent(env, opts, exploration_noise=exploration_noise)

    # train
    agent.train(num_steps)


if __name__ == '__main__':
    main()
