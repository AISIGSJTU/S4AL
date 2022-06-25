import numpy as np
import torch
from typing import Dict


class ReplayBuffer:

    def __init__(self, obs_dim, action_dim, size: int, batch_size: int = 32):
        if isinstance(obs_dim, list) or isinstance(obs_dim, tuple):
            self.obs_buf = torch.zeros([size, *obs_dim], dtype=torch.float32)
            self.next_obs_buf = torch.zeros([size, *obs_dim], dtype=torch.float32)
        elif isinstance(obs_dim, int):
            self.obs_buf = torch.zeros([size, obs_dim], dtype=torch.float32)
            self.next_obs_buf = torch.zeros([size, obs_dim], dtype=torch.float32)
        else:
            raise RuntimeError()
        self.acts_buf = np.zeros([size, action_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def store(
        self,
        obs: torch.tensor,
        act: torch.tensor,
        rew: float,
        next_obs: torch.tensor,
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size
