from albumentations import Compose, IAAPiecewiseAffine, ElasticTransform
import glob
import numpy as np
from PIL import Image
import torch
from typing import Tuple

from TD3.transforms import RotateWProb
from predictor.model import Predictor


class FourthActionNormalizer:
    def __init__(self, act0_range: tuple, act1_range: tuple, act2_range: tuple, act3_range: tuple):
        self.act0_low, self.act0_high = act0_range
        self.act1_low, self.act1_high = act1_range
        self.act2_low, self.act2_high = act2_range
        self.act3_low, self.act3_high = act3_range

        self.act0_scale_factor = (self.act0_high - self.act0_low) / 2
        self.act0_reloc_factor = self.act0_high - self.act0_scale_factor

        self.act1_scale_factor = (self.act1_high - self.act1_low) / 2
        self.act1_reloc_factor = self.act1_high - self.act1_scale_factor

        self.act2_scale_factor = (self.act2_high - self.act2_low) / 2
        self.act2_reloc_factor = self.act2_high - self.act2_scale_factor

        self.act3_scale_factor = (self.act3_high - self.act3_low) / 2
        self.act3_reloc_factor = self.act3_high - self.act3_scale_factor

    def reverse_action(self, action: np.ndarray) -> np.ndarray:
        """
        Change the range (-1, 1) to (low, high).
        """
        reversed_action = np.copy(action)
        reversed_action[0] = reversed_action[0] * self.act0_scale_factor + self.act0_reloc_factor
        reversed_action[0] = np.clip(reversed_action[0], self.act0_low, self.act0_high)
        reversed_action[1] = reversed_action[1] * self.act1_scale_factor + self.act1_reloc_factor
        reversed_action[1] = np.clip(reversed_action[1], self.act1_low, self.act1_high)
        reversed_action[2] = reversed_action[2] * self.act2_scale_factor + self.act2_reloc_factor
        reversed_action[2] = np.clip(reversed_action[2], self.act2_low, self.act2_high)
        reversed_action[3] = reversed_action[3] * self.act3_scale_factor + self.act3_reloc_factor
        reversed_action[3] = np.clip(reversed_action[3], self.act3_low, self.act3_high)
        return reversed_action

    def norm_action(self, action: np.ndarray) -> np.ndarray:
        """
        Change the range (low, high) to (-1, 1).
        """
        normed_action = np.copy(action)
        normed_action[0] = (normed_action[0] - self.act0_reloc_factor) / self.act0_scale_factor
        normed_action[1] = (normed_action[1] - self.act1_reloc_factor) / self.act1_scale_factor
        normed_action[2] = (normed_action[2] - self.act2_reloc_factor) / self.act2_scale_factor
        normed_action[3] = (normed_action[3] - self.act3_reloc_factor) / self.act3_scale_factor
        normed_action = np.clip(normed_action, -1.0, 1.0)
        return normed_action


class FractalDataset:
    def __init__(self, root):
        self.imgs_list = glob.glob("{}/*.png".format(root))
        self.imgs_list.sort()

        self.ptr = 0
        self.max_size = len(self.imgs_list)

    def get_item(self) -> Tuple[np.ndarray, int]:
        img_path = self.imgs_list[self.ptr]
        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img)
        result = (img, self.ptr)
        self.ptr = (self.ptr + 1) % self.max_size
        return result

    def __len__(self):
        return len(self.imgs_list)


class FractalEnv:
    def __init__(self, obs_dim, train_root, action_range: list, device, predictor_ckp):
        # the dataset of raw fractals
        self.dataset = FractalDataset(train_root)

        # build the fid predictor model and load the trained weight
        self.fid_predictor = Predictor(num_classes=1).to(device)
        self.fid_predictor.load_state_dict(torch.load(predictor_ckp)['model_state_dict'])
        self.fid_predictor.eval()

        self.obs_dim = obs_dim
        self.action_dim = 4

        self.action_normalizer = FourthActionNormalizer(
            action_range[0], action_range[1], action_range[2], action_range[3])

        self.img = None
        self.len_traj = 0  # the length of the trajectory
        self.max_traj = 1  # the maximum length of the trajectory during training

    def reset(self):
        self.img, img_ind = self.dataset.get_item()
        self.len_traj = 0
        return self.img, img_ind

    def step(self, action) -> Tuple[np.ndarray, int, bool]:
        # rescale the action from (-1, 1) to the real range
        action = self.action_normalizer.reverse_action(action)

        albu_transform = Compose([
            IAAPiecewiseAffine(
                scale=(action[0], action[0] + 1e-6),
                always_apply=True,
            ),
            ElasticTransform(
                alpha=action[1],
                sigma=action[2],
                always_apply=True,
            ),
        ], p=1)
        rotate_transform = RotateWProb(
            angle=action[3],
            always_apply=True,
        )

        aug_img = albu_transform(image=self.img)['image']
        aug_img = rotate_transform(aug_img)
        self.img = aug_img

        # the calculation of the reward is located in the agent,
        # so as to save some time by mini-batch.
        # Here, we only provide a dummy reward.
        reward = 0

        self.len_traj += 1
        if self.len_traj >= self.max_traj:
            done = True
        else:
            done = False

        return self.img, reward, done
