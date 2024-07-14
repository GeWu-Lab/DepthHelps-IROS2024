import os
import sys
import time
import json
import yaml
import math
import glob
import pickle
import shutil
import logging
import argparse
import functools
from dataclasses import dataclass
from multiprocessing import Value
from typing import Optional, OrderedDict

import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import deepspeed
from tqdm import tqdm
from einops import rearrange
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DataParallel as DP
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

import utils.common as common_utils
from models.roboflamingo.helpers import PerceiverResampler, NoLatentPerceiverResampler
from datasets.depth_libero_dataset import load_DepthLiberoDataset
from datasets.real_demo_dataset import load_RealDemoDataset
PROJECT_DIR = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1])

class VisionEncoder(nn.Module):

    def __init__(
        self, 
        vision_encoder: nn.Module, 
        perceiver: nn.Module,
        depth_sensor_encoder: nn.Module,
        depth_perceiver: nn.Module,
        embedding_dim: int,
        pred_type: str = "resampler",
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.perceiver = perceiver
        self.depth_sensor_encoder = depth_sensor_encoder
        self.depth_perceiver = depth_perceiver
        self.embedding_dim = embedding_dim
        self.pred_type = pred_type
        if pred_type == "resampler":
            self.depth_pred = PerceiverResampler(dim=self.embedding_dim)
        elif pred_type == "linear":
            self.depth_pred = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(self.embedding_dim // 2, self.embedding_dim),
            )
        elif pred_type == "resampler_no_latent":
            print("Using NoLatentPerceiverResampler")
            self.depth_pred = NoLatentPerceiverResampler(dim=self.embedding_dim)
        elif pred_type == "resampler_fix_latent":
            print("Fix Latent")
            self.depth_pred = PerceiverResampler(dim=self.embedding_dim)
            nn.init.constant_(self.depth_pred.latents, 1.0)
            self.depth_pred.latents.requires_grad_(False)
        else:
            raise NotImplementedError

    def _encode_vision(self, vision_x: torch.Tensor):
        """
        Args:
            vision_x (torch.Tensor): shape = (B * window_size, 1, 1, 3, H, W)
        Returns:
            vision_x (torch.Tensor): shape = (B * window_size, 1, 1, patch_num, vision_embedding_dim)
        """
        assert vision_x.ndim == 6, "vision_x should be of shape (B * window_size, 1, 1, 3, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"
        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")  # (B * window_size * 1 * 1, 3, H, W)
        with torch.no_grad():
            vision_x = self.vision_encoder.visual(vision_x)[1]  # (B * window_size * 1 * 1, patch_num, vision_embedding_dim)
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
        return vision_x

    def _encode_depth_sensor(self, depth_sensor: torch.Tensor):
        """
        Args:
            depth_sensor (torch.Tensor): shape = (B * window_size, 1, 1, 3, H, W)
        Returns:
            depth_sensor (torch.Tensor): shape = (B * window_size, 1, 1, patch_num, vision_embedding_dim)
        """
        assert depth_sensor.ndim == 6, "depth_sensor should be of shape (B * window_size, 1, 1, 3, H, W)"
        b, T, F = depth_sensor.shape[:3]
        assert F == 1, "Only single frame supported"
        depth_sensor = rearrange(depth_sensor, "b T F c h w -> (b T F) c h w")  # (B * window_size * 1 * 1, 3, H, W)
        with torch.no_grad():
            depth_sensor = self.depth_sensor_encoder.visual(depth_sensor)[1]  # (B * window_size * 1 * 1, patch_num, vision_embedding_dim)
        depth_sensor = rearrange(depth_sensor, "(b T F) v d -> b T F v d", b=b, T=T, F=F)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
        return depth_sensor

    def forward(
        self, 
        vision_x: torch.Tensor,
        vision_gripper: torch.Tensor,
        depth_static: torch.Tensor, 
        depth_gripper: torch.Tensor,
    ):
        with torch.no_grad():
            if vision_x is not None:
                B = vision_x.shape[0]
                vision_x = vision_x.reshape(-1, *vision_x.shape[2:])
                vision_gripper = vision_gripper.reshape(-1, *vision_gripper.shape[2:])
            if depth_static is not None:
                B = depth_static.shape[0]
                depth_static = depth_static.reshape(-1, *depth_static.shape[2:])
                depth_gripper = depth_gripper.reshape(-1, *depth_gripper.shape[2:])
            
            vision_x = self._encode_vision(vision_x)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
            vision_gripper = self._encode_vision(vision_gripper)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
            vision_x = self.perceiver(vision_x)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
            vision_gripper = self.perceiver(vision_gripper)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64

            depth_static = self._encode_depth_sensor(depth_static)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
            depth_gripper = self._encode_depth_sensor(depth_gripper)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
            depth_static = self.depth_perceiver(depth_static)  # (B * window_size, 1, num_latents, depth_embedding_dim), num_latents = 64
            depth_gripper = self.depth_perceiver(depth_gripper)  # (B * window_size, 1, num_latents, depth_embedding_dim), num_latents = 64

        if self.pred_type == "resampler":
            depth_static_pred = self.depth_pred(vision_x.unsqueeze(dim=1))
            depth_gripper_pred = self.depth_pred(vision_gripper.unsqueeze(dim=1))
        elif self.pred_type == "linear":
            depth_static_pred = self.depth_pred(vision_x)
            depth_gripper_pred = self.depth_pred(vision_gripper)
        elif self.pred_type == "resampler_no_latent":
            depth_static_pred = self.depth_pred(vision_x.unsqueeze(dim=1))
            depth_gripper_pred = self.depth_pred(vision_gripper.unsqueeze(dim=1))
        elif self.pred_type == "resampler_fix_latent":
            depth_static_pred = self.depth_pred(vision_x.unsqueeze(dim=1))
            depth_gripper_pred = self.depth_pred(vision_gripper.unsqueeze(dim=1))

                
        return (F.mse_loss(depth_static_pred, depth_static) + F.mse_loss(depth_gripper_pred, depth_gripper)) / 2

def get_logger(
    filename: str,
    logger_name: Optional[str] = None,
    log_level: int = logging.INFO,
    fmt_str: str = "[%(asctime)s] [%(levelname)s] %(message)s",
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    handler = logging.FileHandler(filename, mode="w")
    handler.setLevel(log_level)  # handler设置日志级别
    handler.setFormatter(logging.Formatter(fmt_str))  # handler对象设置格式
    logger.addHandler(handler)
    return logger

def load_dataset(image_processor, dataset_name):
    if dataset_name == "DepthLiberoDataset":
        dataset_config = {
            "load_fn": "load_DepthLiberoDataset",
            "dataset_dir": "data/LIBERO/v1",
            "window_size": 12,
            "pad": True,
            "rgb_pad": 10,
            "gripper_pad": 4,
            "use_ceph": True,  # whether training in the S cluster
            "used_classes": ["libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial"],
            "metadata_file": "metadata/libero.txt",
        }
        def _preprocess_image(sample, image_processor):
            """
            sample: [<class 'PIL.Image.Image'>, ...]
            Return: Tensor shape = (seq_len, 3, 224, 224)
            """
            image = [image_processor(s).unsqueeze(0) for s in sample]  # [(1, 3, 224, 224), ...]
            image = torch.cat(image, dim=0)  # (seq_len, 3, 224, 224)
            return image
        def _preprocess_text(sample, text_tokenizer):
            max_token = max([len(s) for s in sample])
            return {
                "input_ids": torch.randint(0, 50, (len(sample), max_token)), 
                "attention_mask": torch.randint(1, 2, (len(sample), max_token))
            }

        dataset = load_DepthLiberoDataset(
            image_fn=functools.partial(_preprocess_image, image_processor=image_processor),
            text_fn=functools.partial(_preprocess_text, text_tokenizer=None), 
            dataset_dir=dataset_config["dataset_dir"],
            window_size=dataset_config["window_size"],
            pad=dataset_config["pad"],
            rgb_pad=dataset_config["rgb_pad"],
            gripper_pad=dataset_config["gripper_pad"],
            use_ceph=dataset_config["use_ceph"],
            used_classes=dataset_config["used_classes"],
            metadata_file=dataset_config["metadata_file"],
        )
        return dataset
    elif dataset_name == "DepthRealDemoDataset":
        dataset_config = {
            "load_fn": "load_RealDemoDataset",
            "return_depth": True,
            "dataset_dir": "data/real_demo",
            "window_size": 12,
            "pad": True,
            "rgb_pad": 10,
            "gripper_pad": 4,
            "use_ceph": False,  # whether training in the S cluster
            "metadata_file": "metadata/real_demo.txt",
        }

        def _preprocess_image(sample, image_processor):
            """
            sample: [<class 'PIL.Image.Image'>, ...]
            Return: Tensor shape = (seq_len, 3, 224, 224)
            """
            image = [image_processor(s).unsqueeze(0) for s in sample]  # [(1, 3, 224, 224), ...]
            image = torch.cat(image, dim=0)
            return image
        def _preprocess_text(sample, text_tokenizer):
            max_token = max([len(s) for s in sample])
            return {
                "input_ids": torch.randint(0, 50, (len(sample), max_token)), 
                "attention_mask": torch.randint(1, 2, (len(sample), max_token))
            }
        dataset = load_RealDemoDataset(
            return_depth=dataset_config["return_depth"],
            image_fn=functools.partial(_preprocess_image, image_processor=image_processor),
            text_fn=functools.partial(_preprocess_text, text_tokenizer=None), 
            dataset_dir=dataset_config["dataset_dir"],
            window_size=dataset_config["window_size"],
            pad=dataset_config["pad"],
            rgb_pad=dataset_config["rgb_pad"],
            gripper_pad=dataset_config["gripper_pad"],
            use_ceph=dataset_config["use_ceph"],
            metadata_file=dataset_config["metadata_file"],
        )
        return dataset

def load_model(pred_type: str = "resampler"):
    vision_encoder_path = "ViT-L-14"
    vision_encoder_pretrained = "openai"
    vision_encoder_cache_dir = f"{PROJECT_DIR}/RoboFlamingo/models/clip"
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        model_name=vision_encoder_path,
        pretrained=vision_encoder_pretrained,
        cache_dir=vision_encoder_cache_dir
    )
    vision_encoder.visual.output_tokens = True

    rgb_perceiver = PerceiverResampler(
        dim=open_clip.get_model_config(vision_encoder_path)["vision_cfg"]["width"]
    )

    depth_sensor_encoder = vision_encoder

    depth_perceiver = PerceiverResampler(
        dim=open_clip.get_model_config(vision_encoder_path)["vision_cfg"]["width"]
    )

    model = VisionEncoder(
        vision_encoder=vision_encoder, 
        perceiver=rgb_perceiver,
        depth_sensor_encoder=depth_sensor_encoder,
        depth_perceiver=depth_perceiver,
        embedding_dim=open_clip.get_model_config(vision_encoder_path)["vision_cfg"]["width"],
        pred_type=pred_type,
    )
    model.requires_grad_(False)
    model.depth_pred.requires_grad_(True)

    return model, image_processor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--backend", type=str, default="nccl")

    parser.add_argument("--run_name", type=str, default="runs/dummy")
    parser.add_argument("--checkpoint", type=str, default="")

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--num_epochs", type=int, default=5)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--steps_per_print", type=int, default=100)
    parser.add_argument("--wall_clock_breakdown", type=bool, default=False)

    parser.add_argument("--auto_remove_prev_ckpt", action="store_true")

    parser.add_argument("--pred_type", type=str, default="resampler", choices=["resampler", "linear", "resampler_no_latent", "resampler_fix_latent"])
    parser.add_argument("--dataset_name", type=str, default="DepthLiberoDataset")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()

    if os.environ.get("MULTINODE_MULTIGPU", "0") == "1":
        assert "DEEPSPEED_PORT" in os.environ and "DEEPSPEED_ADDR" in os.environ
        os.environ['RANK'] = os.environ['SLURM_PROCID']
        os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
        os.environ["MASTER_ADDR"] = os.environ["DEEPSPEED_ADDR"]
        os.environ['MASTER_PORT'] = os.environ['DEEPSPEED_PORT']
        os.environ['LOCAL_RANK'] = os.environ['SLURM_LOCALID']
        os.environ["NCCL_IB_GID_INDEX"] = "0"
        os.environ["NCCL_IB_HCA"] = "mlx5_0"
        os.environ['NCCL_SOCKET_IFNAME'] = "eth0"
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        import subprocess
        p = subprocess.run("ibv_devinfo", shell=True, stdout=subprocess.PIPE)
        ibv_devinfo = p.stdout.decode()
        p = subprocess.run("ifconfig", shell=True, stdout=subprocess.PIPE)
        ifconfig = p.stdout.decode()
        p = subprocess.run("hostname -I", shell=True, stdout=subprocess.PIPE)
        hostname = p.stdout.decode()
        print(f"=> [info] [{hostname.strip()}] ibv_devinfo \n{ibv_devinfo}")
        print(f"=> [info] [{hostname.strip()}] ifconfig \n{ifconfig}")
        print(f"=> [info] [{hostname.strip()}] NCCL_SOCKET_IFNAME {os.environ['NCCL_SOCKET_IFNAME']}")
        print(f"=> [info] [{hostname.strip()}] NCCL_IB_HCA {os.environ['NCCL_IB_HCA']}")
        print(f"=> [info] [{hostname.strip()}] rank {os.environ['RANK']}")
        print(f"=> [info] [{hostname.strip()}] world size {os.environ['WORLD_SIZE']}")
        print(f"=> [info] [{hostname.strip()}] master addr {os.environ['MASTER_ADDR']}")
        print(f"=> [info] [{hostname.strip()}] master port {os.environ['MASTER_PORT']}")
        print(f"=> [info] [{hostname.strip()}] local rank {os.environ['LOCAL_RANK']}")
        deepspeed.init_distributed(dist_backend=args.backend)
        os.makedirs(args.run_name, exist_ok=True)
        os.makedirs(f"{args.run_name}/ckpt", exist_ok=True)
        logger = get_logger(
            filename=f"{args.run_name}/train.log",
            logger_name="roboflamingo",
        )
        print(args)
        with open(f"{args.run_name}/cmd.txt", "w") as f:
            f.write("python " + " ".join(sys.argv) + "\n")
            f.write(json.dumps(vars(args), indent=4) + "\n")
    else:
        deepspeed.init_distributed(dist_backend=args.backend)
        os.makedirs(args.run_name, exist_ok=True)
        os.makedirs(f"{args.run_name}/ckpt", exist_ok=True)
        if os.path.exists(f"{args.run_name}/train.log"):
            log_names = glob.glob(f"{args.run_name}/train-*.log")
            log_idx = 0
            if len(log_names) > 0:
                log_names.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))
                log_idx = int(log_names[-1].split("-")[-1].split(".")[0]) + 1
            if args.local_rank == 0:
                logger = get_logger(
                    filename=f"{args.run_name}/train-{log_idx}.log",
                    logger_name="roboflamingo",
                )
            else:
                logger = get_logger(
                    filename=f"{args.run_name}/None-train.log",
                    logger_name="roboflamingo",
                )
        else:  
            logger = get_logger(
                filename=f"{args.run_name}/train.log",
                logger_name="roboflamingo",
            )
        print(args)
        with open(f"{args.run_name}/cmd.txt", "w") as f:
            f.write("python " + " ".join(sys.argv) + "\n")
            f.write(json.dumps(vars(args), indent=4) + "\n")

    model, image_processor = load_model(args.pred_type)

    if args.local_rank == 0:
        print(
            f"model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
        )
        logger.info(
            f"model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
        )

    if args.checkpoint != "" and args.checkpoint != "none":
        args.checkpoint = args.checkpoint.split("#")
        for checkpoint in args.checkpoint:
            if args.local_rank == 0: logger.info(f"load checkpoint from {checkpoint}")
            ckpt = torch.load(checkpoint, map_location="cpu")
            if "module" in ckpt.keys():
                assert "model_state_dict" not in ckpt["module"].keys()
                missing_keys, unexpected_keys = model.load_state_dict(ckpt["module"], strict=False)
                if args.local_rank == 0:
                    print(f"missing_keys: {missing_keys}")
                    print(f"unexpected_keys: {unexpected_keys}")
                    logger.info(f"missing keys: {missing_keys}")
                    logger.info(f"unexpected keys: {unexpected_keys}")
            else:
                raise NotImplementedError
    train_dataset = load_dataset(image_processor, args.dataset_name)
    
    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return (
                "gated_cross_attn_layer" in x
                and "ff_gate" not in x
                and "attn_gate" not in x
                and "norm" not in x
                and "bias" not in x
            )

        for n, p in model.named_parameters():
            
            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": [p for p in params_with_wd if p.requires_grad], "weight_decay": args.weight_decay},
            {"params": [p for p in params_without_wd if p.requires_grad], "weight_decay": 0.0},
        ]
    args.learning_rate = args.learning_rate # adaptive lr
    optimizer = torch.optim.AdamW(get_grouped_params(model), lr=args.learning_rate)
    
    lr_scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps
    )
    
    config={
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": args.train_micro_batch_size_per_gpu,
        "steps_per_print": args.steps_per_print,
        "wall_clock_breakdown": args.wall_clock_breakdown
    }
    engine, _, dataloader, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        optimizer=optimizer,
        training_data=train_dataset,
        lr_scheduler=lr_scheduler,
        collate_fn=train_dataset.collater,
        config=config
    )
    engine.load_checkpoint(f"{args.run_name}/ckpt")
    step = engine.global_steps
    resume_epoch = engine.global_steps // len(dataloader) # resume training from the checkpoint loaded
    if args.local_rank == 0:
        logger.info(f"resume training from epoch {resume_epoch}")
        print(f"resume training from epoch {resume_epoch}")

    for epoch in range(resume_epoch, args.num_epochs):
        # setup logging
        step_time_m = common_utils.AverageMeter()
        data_time_m = common_utils.AverageMeter()  
        end = time.time()
        t = tqdm(
            enumerate(dataloader),
            disable=engine.local_rank != 0,
            total=len(dataloader),
        )
        t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
        t1 = time.time()
        time_avgmeter = common_utils.AverageMeter()
        for num_steps, batch in t:
            assert type(batch) == dict
            vision_x = batch["image_tensors"]
            vision_gripper = batch["gripper_tensors"]
            depth_static = batch["depth_static_tensors"]
            depth_gripper = batch["depth_gripper_tensors"]

            vision_x = (vision_x.to(engine.local_rank, non_blocking=True).unsqueeze(2).unsqueeze(2))
            vision_gripper = (vision_gripper.to(engine.local_rank, non_blocking=True).unsqueeze(2).unsqueeze(2))  # (B, window_size, 1, 1, 3, 224, 224)
            depth_static = (depth_static.to(engine.local_rank, non_blocking=True).unsqueeze(2).unsqueeze(2))
            depth_gripper = (depth_gripper.to(engine.local_rank, non_blocking=True).unsqueeze(2).unsqueeze(2))  # (B, window_size, 1, 1, 3, 224, 224)

            loss = engine(vision_x, vision_gripper, depth_static, depth_gripper)
            engine.backward(loss)
            engine.step()

            if num_steps % args.gradient_accumulation_steps == 0:
                step += 1
                if step % args.steps_per_print == 0 and args.local_rank == 0:
                    print(f"epoch {epoch+1}/{args.num_epochs} step {num_steps}/{len(dataloader)} loss {loss.item()}")
                if step % 10 == 0 and engine.local_rank == 0:
                    time_avgmeter.update(time.time() - t1)
                    t1 = time.time()
                    logger.info(f"step: {step:3d} loss: {loss} time: {time_avgmeter.avg}")
        prev_ckpt_name = None
        if os.path.exists(f"{args.run_name}/ckpt/latest"):
            with open(f"{args.run_name}/ckpt/latest", "r") as fi: prev_ckpt_name = fi.read().strip()
        engine.save_checkpoint(f"{args.run_name}/ckpt")
        if args.auto_remove_prev_ckpt:
            if prev_ckpt_name is not None:
                if engine.local_rank == 0:
                    if os.path.exists(f"{args.run_name}/ckpt/{prev_ckpt_name}"):
                        try:
                            if os.path.islink(f"{args.run_name}/ckpt/{prev_ckpt_name}"):
                                os.remove(f"{args.run_name}/ckpt/{prev_ckpt_name}")
                            else:
                                shutil.rmtree(f"{args.run_name}/ckpt/{prev_ckpt_name}")
                        except Exception as e:
                            if os.environ.get("MULTINODE_MULTIGPU", "0") == "1":
                                logger.info(f' failed to remove {args.run_name}/ckpt/{prev_ckpt_name} hostname = {hostname} masterhost = {os.environ["DEEPSPEED_ADDR"]}')
                            else:
                                raise e
                            
if __name__ == "__main__":
    main()
