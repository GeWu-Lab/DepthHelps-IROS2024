import os
import io
from pathlib import Path
from typing import List, Callable

import cv2
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def load_h5py(path: Path):
    return h5py.File(path, "r")

def build_indices(dataset_dir: Path, window_size: int, tasks: List[str] = None, skip_size: int=1):
    episode_lookup = []
    for task in tasks:
        sub_tasks = filter(lambda x: x != ".DS_Store", os.listdir(dataset_dir / task))
        for sub_task in sub_tasks:
            print(task, sub_task)
            with h5py.File(dataset_dir / task / sub_task, "r") as f:
                demo_ids = list(f["data"].keys())
                for demo_id in demo_ids:
                    episode_length = f["data"][demo_id]["actions"].shape[0]
                    assert episode_length >= window_size
                    for idx in range(0, episode_length - window_size, skip_size):
                        episode_lookup.append((task, sub_task, demo_id, idx))
    return episode_lookup
if os.environ.get("BUILD_INDICES", "0") == "1":
    episode_lookup = build_indices(Path("data/LIBERO/v1"), 12, ["libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial"], skip_size=6)
    with open("metadata/libero.txt", "w") as fo:
        for idx in range(len(episode_lookup)):
            fo.write(episode_lookup[idx][0] + " " + episode_lookup[idx][1] + " " + episode_lookup[idx][2] + " " + str(episode_lookup[idx][3]) + "\n")

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)

    def forward_traj(self, x):
        n, t, c, h, w = x.size()
        x = x.view(n*t, *x.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n*t, *base_grid.shape[2:])
        shift = torch.randint(1,
                              2 * self.pad + 1,
                              size=(n*t, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        return x

    def forward_traj_with_depth(self, x, depth):
        n, t, c, h, w = x.size()
        x = x.view(n*t, *x.shape[2:])
        depth = depth.view(n*t, *depth.shape[2:])
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        depth = F.pad(depth, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        base_grid = base_grid.unsqueeze(1).repeat(1, t, 1, 1, 1)
        base_grid = base_grid.view(n*t, *base_grid.shape[2:])
        shift = torch.randint(1,
                              2 * self.pad + 1,
                              size=(n*t, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        x = F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
        depth = F.grid_sample(depth, grid, padding_mode='zeros', align_corners=False)
        x = x.view(n, t, *x.shape[1:])
        depth = depth.view(n, t, *depth.shape[1:])
        return x, depth

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
class DepthNorm(nn.Module):
    def __init__(
        self,
        max_depth=0,
        min_depth=0.01,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.min_depth = min_depth

    def forward(self, depth_img):
        # image = np.array(image)
        depth_img = depth_img.clip(min=self.min_depth)
        if self.max_depth != 0:
            depth_img = depth_img.clip(max=self.max_depth)
            depth_img /= self.max_depth   #  0-1
        else:
            depth_img /= depth_img.max()
        depth_img = torch.from_numpy(depth_img).unsqueeze(0).repeat(3, 1, 1)  # assume image
        return depth_img.to(torch.get_default_dtype())

class DepthLiberoDataset(Dataset):

    def __init__(
        self,
        image_fn: Callable,
        text_fn: Callable,
        dataset_dir: str = "data/LIBERO/v1",
        window_size: int = 12,
        pad: bool = True,
        rgb_pad: int=10,
        gripper_pad: int=4,
        use_ceph: bool = False,  # whether training in the S cluster
        used_classes: List[str] = None,
        metadata_file: str = "metadata/libero.txt"
    ) -> None:
        self.image_fn = image_fn
        self.text_fn = text_fn
        self.dataset_dir: Path = Path(dataset_dir)
        self.used_classes: List[str] = used_classes
        self.use_ceph: bool = use_ceph
        self.window_size: int = window_size
        self.pad: bool = pad
        self.rgb_pad: int = rgb_pad
        if self.rgb_pad != -1:
            self.rgb_shift = RandomShiftsAug(rgb_pad)
        self.gripper_pad: int = gripper_pad
        if self.gripper_pad != -1:
            self.gripper_shift = RandomShiftsAug(gripper_pad)
        with open(metadata_file, "r") as fi: episode_lookup = fi.read().splitlines()
        self.episode_lookup = [x.split(" ") for x in episode_lookup]

        self.depth_transform = transforms.Compose(
            [
                DepthNorm(max_depth=5.0),
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),  # assume image
            ]
        )
        def _depth_fn(sample):
            image = [self.depth_transform(s).unsqueeze(0) for s in sample]  # [(1, 3, 224, 224), ...]
            image = torch.cat(image, dim=0)
            return image

        self.depth_fn = _depth_fn
    
    def __len__(self):
        return len(self.episode_lookup)
    
    def __getitem__(self, idx):
        episode = self.episode_lookup[idx]
        task, sub_task, demo_id, start_idx = episode
        start_idx = int(start_idx)
        f = load_h5py(self.dataset_dir / task / sub_task)
        actions = f["data"][demo_id]["actions"][start_idx:start_idx+self.window_size]
        rgb_static = f["data"][demo_id]["obs"]["agentview_rgb"][start_idx:start_idx+self.window_size]
        rgb_gripper = f["data"][demo_id]["obs"]["eye_in_hand_rgb"][start_idx:start_idx+self.window_size]
        language = " ".join(sub_task.split("_")[:-1])
        f.close()
        f_depth = load_h5py(self.dataset_dir / "depth" / task / sub_task)
        depth_static = f_depth["data"][demo_id]["obs"]["agentview_depth"][start_idx:start_idx+self.window_size]
        depth_gripper = f_depth["data"][demo_id]["obs"]["eye_in_hand_depth"][start_idx:start_idx+self.window_size]
        f_depth.close()
        return {
            "actions": actions,
            "rgb_gripper": [Image.fromarray(img) for img in rgb_gripper],
            "rgb_static": [Image.fromarray(img) for img in rgb_static],
            "depth_gripper": depth_gripper[:, :, :, 0],
            "depth_static": depth_static[:, :, :, 0],
            "language": language
        }
    
    def collater(self, sample):
        action_tensors = torch.from_numpy(np.stack([x["actions"] for x in sample])).float()
        gripper_tensors = torch.stack([
            self.image_fn(x["rgb_gripper"]) 
            for x in sample
        ])
        image_tensors = torch.stack([
            self.image_fn(x["rgb_static"]) 
            for x in sample
        ])
        depth_gripper_tensors = torch.stack([
            self.depth_fn(x["depth_gripper"]) 
            for x in sample
        ])
        depth_static_tensors = torch.stack([
            self.depth_fn(x["depth_static"]) 
            for x in sample
        ])
        stacked_language = [x["language"] for x in sample]
        text_tensors, attention_mask = self.text_fn(stacked_language)

        if self.rgb_pad != -1:
            image_tensors, depth_static_tensors = self.rgb_shift.forward_traj_with_depth(image_tensors, depth_static_tensors)
        if self.gripper_pad != -1:
            gripper_tensors, depth_gripper_tensors = self.gripper_shift.forward_traj_with_depth(gripper_tensors, depth_gripper_tensors)
        
        # return (image_tensors, gripper_tensors), action_tensors, (text_tensors, attention_mask)
        return {
            "image_tensors": image_tensors,
            "gripper_tensors": gripper_tensors,
            "depth_static_tensors": depth_static_tensors,
            "depth_gripper_tensors": depth_gripper_tensors,
            "action_tensors": action_tensors,
            "text_tensors": text_tensors,
            "attention_mask": attention_mask
        }

def load_DepthLiberoDataset(
    image_fn: Callable,
    text_fn: Callable,
    dataset_dir: str = "data/LIBERO/v1",
    window_size: int = 12,
    pad: bool = True,
    rgb_pad: int=10,
    gripper_pad: int=4,
    use_ceph: bool = False,  # whether training in the S cluster
    used_classes: List[str] = None,
    metadata_file: str = "metadata/libero.txt",
):
    return DepthLiberoDataset(
        image_fn=image_fn,
        text_fn=text_fn,
        dataset_dir=dataset_dir,
        window_size=window_size,
        pad=pad,
        rgb_pad=rgb_pad,
        gripper_pad=gripper_pad,
        use_ceph=use_ceph,
        used_classes=used_classes,
        metadata_file=metadata_file,
    )
