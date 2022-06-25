import numpy as np
import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.transforms as TF
from typing import Tuple, Dict

from TD3.replay_buffer import ReplayBuffer
from TD3.noise import GaussianNoise
from TD3.actor import Actor
from TD3.critic import Critic
from TD3.feature_encoder import FeatureEncoder
from TD3.env import FractalEnv

np.set_printoptions(precision=3)


class TD3Agent:
    """
    TD3Agent interacting with environment.

    Attribute:
        env (FractalEnv): environment
        actor (nn.Module): actor model
        actor_target (nn.Module): target actor model
        actor_optimizer (Optimizer): optimizer for training actor
        critic1 (nn.Module): critic model
        critic2 (nn.Module): critic model
        critic_target1 (nn.Module): target critic model
        critic_target2 (nn.Module): target critic model
        critic_optimizer (Optimizer): optimizer for training critic
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        gamma (float): discount factor
        tau (float): parameter for soft target update
        initial_random_steps (int): initial random action steps
        exploration_noise (GaussianNoise): gaussian noise for policy
        target_policy_noise (GaussianNoise): gaussian noise for target policy
        target_policy_noise_clip (float): clip target gaussian noise
        device (torch.device): cpu / gpu
        transition (list): temporory storage for the recent transition
        policy_update_freq (int): update actor every time critic updates this times
        total_step (int): total step numbers
        is_test (bool): flag to show the current mode (train / test)
    """

    def __init__(
            self,
            env: FractalEnv,
            opts: dict,
            tau: float = 5e-3,
            policy_update_freq: int = 2,
            exploration_noise: Dict = None,
            target_policy_noise: float = 0.2,
            target_policy_noise_clip: float = 0.5,
    ):
        self.obs_dim = env.obs_dim
        self.action_dim = env.action_dim

        self.device = opts['device']

        # feature encoder
        self.encoder = FeatureEncoder().to(self.device)
        state_feat_dim = self.encoder.state_feat_dim
        if opts['encoder_path']:
            self.encoder.encoder_load_state_dict(torch.load(opts['encoder_path']))
            print("Load the encoder weight from {}".format(opts['encoder_path']))
        self.encoder.eval()

        self.env = env
        self.memory = ReplayBuffer(state_feat_dim, self.action_dim, opts['memory_size'], opts['batch_size'])
        self.batch_size = opts['batch_size']
        self.gamma = opts['gamma']
        self.initial_random_steps = opts['initial_random_steps']
        self.lr_actor = opts['lr_actor']
        self.lr_critic = opts['lr_critic']
        self.tau = tau
        self.policy_update_freq = policy_update_freq

        # noise
        self.exploration_noise = GaussianNoise(
            self.action_dim, exploration_noise['min_sigma'],
            exploration_noise['max_sigma'], exploration_noise['decay_period'],
        )
        self.target_policy_noise = GaussianNoise(
            self.action_dim, target_policy_noise, target_policy_noise
        )
        self.target_policy_noise_clip = target_policy_noise_clip

        # networks
        self.actor = Actor(self.action_dim, state_feat_dim).to(self.device)
        self.actor_target = Actor(self.action_dim, state_feat_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = Critic(self.action_dim, state_feat_dim).to(self.device)
        self.critic_target1 = Critic(self.action_dim, state_feat_dim).to(self.device)
        self.critic_target1.load_state_dict(self.critic1.state_dict())

        self.critic2 = Critic(self.action_dim, state_feat_dim).to(self.device)
        self.critic_target2 = Critic(self.action_dim, state_feat_dim).to(self.device)
        self.critic_target2.load_state_dict(self.critic2.state_dict())

        # concat critic parameters to use one optim
        critic_parameters = list(self.critic1.parameters()) + \
                            list(self.critic2.parameters())

        # optimizer
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = Adam(critic_parameters, lr=self.lr_critic)

        self.init_step = 1

        # transform
        self.transforms = TF.Compose([
            TF.Resize(224),
            TF.ToTensor(),
        ])

        # transitions to store in the memory
        self.transitions = list()
        self.single_transition = list()

        # total steps count
        self.total_step = 0

        # total episode count
        self.total_episode = 0

        # update step for actor
        self.update_step = 0

        # mode: train / test
        self.is_test = False

        # model save path
        self.ckp_path = opts['ckp_path']
        os.makedirs(self.ckp_path, exist_ok=True)

        self.img_ind = 0

    def select_action(self, state: np.ndarray, log_info=False) -> np.ndarray:
        """ Select an action from the input state. """
        state_tensor = self.transforms(Image.fromarray(state))  # numpy array --> PIL Image --> transform
        state_feat = self.encoder(state_tensor.unsqueeze(0).to(self.device)).detach()  # state --> feature
        # if initial random action should be conducted
        if self.total_step < self.initial_random_steps and not self.is_test:
            selected_action = np.random.uniform(low=-1, high=1, size=self.action_dim)
        else:
            selected_action = (self.actor(state_feat)[0].detach().cpu().numpy())

        if log_info:
            print("[log info] ----- Action: {}".format(selected_action))

        # add noise for exploration during training
        if not self.is_test:
            noise, noise_sigma = self.exploration_noise.sample(self.total_step)
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)

            self.single_transition = [
                state_tensor,
                state_feat,
                selected_action
            ]

            if log_info:
                print("[log info] ----- Action + Noise ({:3f}): {}".format(noise_sigma, selected_action))

        return selected_action

    def step(self, action: np.ndarray, log_info=False) -> Tuple[np.ndarray, int, bool, dict]:
        """ Take an action and return the response of the env. """
        next_state, reward, done = self.env.step(action)

        next_state_tensor = self.transforms(Image.fromarray(next_state))  # numpy array --> PIL Image --> transform
        next_state_feat = self.encoder(next_state_tensor.unsqueeze(0).to(self.device)).detach()  # state --> feature

        if not self.is_test:
            self.single_transition += [
                reward,
                next_state_tensor,
                next_state_feat,
                done,
            ]
            self.transitions.append(self.single_transition)

        if done:
            # calculate and update the real rewards
            obs_list = list()
            for i, item in enumerate(self.transitions):
                if i == 0:
                    obs_list.append(item[0])
                obs_list.append(item[4])
            obs_list = torch.stack(obs_list, dim=0).to(self.device)
            fid_list = self.env.fid_predictor(obs_list).detach().cpu().numpy().flatten()
            real_rewards = -(fid_list[1:] - fid_list[:-1])

            info = {
                'reward': np.mean(real_rewards),
                'first_reward': real_rewards[0],
            }

            for i, item in enumerate(self.transitions):
                item.pop(0)  # pop the state_tensor
                item.pop(3)  # pop the next_state_tensor
                item[2] = real_rewards[i]
                self.memory.store(*item)
            del self.transitions[:]

            if log_info:
                print("[log info] ----- Rewards: {}".format(real_rewards))
        else:
            info = {}

        return next_state, reward, done, info

    def update_model(self) -> Tuple[float, float]:
        """ Update the model by gradient descent. """
        device = self.device  # for shortening the following lines

        samples = self.memory.sample_batch()
        states = samples["obs"].to(device)
        next_states = samples["next_obs"].to(device)
        actions = torch.from_numpy(samples["acts"].reshape(-1, self.action_dim)).float().to(device)
        rewards = torch.from_numpy(samples["rews"].reshape(-1, 1)).float().to(device)
        dones = torch.from_numpy(samples["done"].reshape(-1, 1)).float().to(device)
        masks = 1 - dones

        # get actions with noise
        noise = torch.from_numpy(self.target_policy_noise.sample()[0]).float().to(device)
        clipped_noise = torch.clamp(
            noise, -self.target_policy_noise_clip, self.target_policy_noise_clip
        )
        next_actions = (self.actor_target(next_states) + clipped_noise).clamp(-1.0, 1.0)

        # min (Q_1', Q_2')
        next_values1 = self.critic_target1(next_states, next_actions)
        next_values2 = self.critic_target2(next_states, next_actions)
        next_values = torch.min(next_values1, next_values2)

        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_returns = rewards + self.gamma * next_values * masks
        curr_returns = curr_returns.detach()

        # critic loss
        values1 = self.critic1(states, actions)
        values2 = self.critic2(states, actions)
        critic1_loss = F.mse_loss(values1, curr_returns)
        critic2_loss = F.mse_loss(values2, curr_returns)

        # train critic
        critic_loss = critic1_loss + critic2_loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_step % self.policy_update_freq == 0:
            # train actor
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # target update
            self._target_soft_update()
        else:
            actor_loss = torch.zeros(1)

        return actor_loss.item(), critic_loss.item()

    def train(self, num_steps: int):
        """ Train the agent. """
        self.is_test = False

        actor_losses = []
        critic_losses = []
        running_rewards, first_rewards = [], []
        period_running_rewards, period_first_rewards = [], []

        state, self.img_ind = self.env.reset()
        for self.total_step in range(self.init_step, num_steps + 1):

            log_info = True if self.total_episode % 50 == 0 else False

            action = self.select_action(state, log_info)

            next_state, reward, done, info = self.step(action, log_info)

            state = next_state

            # if episode ends
            if done:
                self.total_episode += 1
                running_rewards.append(info['reward'])
                first_rewards.append(info['first_reward'])
                if len(running_rewards) > 1000:
                    running_rewards.pop(0)
                    first_rewards.pop(0)

                # new episode
                state, self.img_ind = self.env.reset()

            # if training is ready
            if len(self.memory) >= self.batch_size and \
                    self.total_step > self.initial_random_steps:
                actor_loss, critic_loss = self.update_model()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

                if self.total_step % 20 == 0:
                    avg_rew = sum(running_rewards) / len(running_rewards)
                    print("[{}] reward {:.3f} actor_loss {:.3f} critic_losses {:.3f}".format(
                        self.total_step, avg_rew, actor_loss, critic_loss))

                    # Save model
                    if self.total_step % 2000 == 0:
                        self.save_model(avg_rew, self.total_step)

            else:
                print("Collecting transitions [{}/{}]".format(self.total_step, self.initial_random_steps))

            if self.total_step % 200 == 0 and self.total_step > self.initial_random_steps:
                period_running_rewards.append(sum(running_rewards) / len(running_rewards))
                period_first_rewards.append(sum(first_rewards) / len(first_rewards))

            if self.total_step % 5000 == 0 and self.total_step > self.initial_random_steps:
                np.savetxt("logs/period_running_rewards.out",
                           np.asarray(period_running_rewards), fmt='%.3f')
                np.savetxt("logs/period_first_rewards.out",
                           np.asarray(period_first_rewards), fmt='%.3f')

    def _target_soft_update(self):
        """ Soft-update: target = tau*local + (1-tau)*target. """
        tau = self.tau
        for t_param, l_param in zip(
                self.actor_target.parameters(), self.actor.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target1.parameters(), self.critic1.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

        for t_param, l_param in zip(
                self.critic_target2.parameters(), self.critic2.parameters()
        ):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

    def save_model(self, avg_rew, step):
        save_name = "{}/ckp_{}.tar".format(self.ckp_path, step)
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic1.state_dict(),
            'critic2': self.critic2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'avg_rew': avg_rew,
            'step': step,
        }, save_name)
