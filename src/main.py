import os
import sys
import time
import json
import glob
import shutil
import logging
import argparse
import functools
from typing import Optional, OrderedDict

import torch
import numpy as np
import torch.nn.functional as F
import deepspeed
from tqdm import tqdm
from transformers import (
    get_constant_schedule_with_warmup,
)

import utils.common as common_utils
from datasets.libero_dataset import load_LiberoDataset
from datasets.depth_libero_dataset import load_DepthLiberoDataset
from models.roboflamingo.flamingo_mpt import load_MPTFlamingo
from models.roboflamingo.flamingo_mpt_depth import load_DepthMPTFlamingo
PROJECT_DIR = "/".join(os.path.dirname(os.path.abspath(__file__)).split("/")[:-1])
MODEL_CONFIG = {
    "roboflamingo": {
        "mpt_3b": {
            "load_fn": "load_MPTFlamingo", 
            "lang_encoder_path": f"{PROJECT_DIR}/RoboFlamingo/models/mpt-1b-redpajama-200b",
            "vision_encoder_path": "ViT-L-14",
            "vision_encoder_pretrained": "openai",
            "vision_encoder_cache_dir": f"{PROJECT_DIR}/RoboFlamingo/models/clip",
            "cross_attn_every_n_layers": 1,
            "openflamingo_checkpoint": f"{PROJECT_DIR}/RoboFlamingo/models/open_flamingo/checkpoint.pt",
            "decoder_layers_attr_name": "transformer.blocks",
            "window_size":12,  # must be the same as DATASET_CONFIG
            "extra_config": {}
        },
        "mpt_3b_KD": {
            "load_fn": "load_DepthMPTFlamingo", 
            "lang_encoder_path": f"{PROJECT_DIR}/RoboFlamingo/models/mpt-1b-redpajama-200b",
            "vision_encoder_path": "ViT-L-14",
            "vision_encoder_pretrained": "openai",
            "vision_encoder_cache_dir": f"{PROJECT_DIR}/RoboFlamingo/models/clip",
            "cross_attn_every_n_layers": 1,
            "openflamingo_checkpoint": f"{PROJECT_DIR}/RoboFlamingo/models/open_flamingo/checkpoint.pt",
            "decoder_layers_attr_name": "transformer.blocks",
            "window_size":12,  # must be the same as DATASET_CONFIG
            "resampler_type": "KD",
            "extra_config": {}
        },
        "mpt_3b_depth": {
            "load_fn": "load_DepthMPTFlamingo", 
            "lang_encoder_path": f"{PROJECT_DIR}/RoboFlamingo/models/mpt-1b-redpajama-200b",
            "vision_encoder_path": "ViT-L-14",
            "vision_encoder_pretrained": "openai",
            "vision_encoder_cache_dir": f"{PROJECT_DIR}/RoboFlamingo/models/clip",
            "cross_attn_every_n_layers": 1,
            "openflamingo_checkpoint": f"{PROJECT_DIR}/RoboFlamingo/models/open_flamingo/checkpoint.pt",
            "decoder_layers_attr_name": "transformer.blocks",
            "window_size":12,  # must be the same as DATASET_CONFIG
            "extra_config": {}
        },
        "mpt_3b_depth_pred_depth": {
            "load_fn": "load_DepthMPTFlamingo", 
            "lang_encoder_path": f"{PROJECT_DIR}/RoboFlamingo/models/mpt-1b-redpajama-200b",
            "vision_encoder_path": "ViT-L-14",
            "vision_encoder_pretrained": "openai",
            "vision_encoder_cache_dir": f"{PROJECT_DIR}/RoboFlamingo/models/clip",
            "cross_attn_every_n_layers": 1,
            "openflamingo_checkpoint": f"{PROJECT_DIR}/RoboFlamingo/models/open_flamingo/checkpoint.pt",
            "decoder_layers_attr_name": "transformer.blocks",
            "window_size":12,  # must be the same as DATASET_CONFIG
            "resampler_type": "pred_depth",  # 这里的resampler_type不会去修改模型的resampler_type，只是用于区分不同的模型
            "extra_config": {}
        },
        "mpt_3b_depth_depth_codebook": {
            "load_fn": "load_DepthMPTFlamingo", 
            "lang_encoder_path": f"{PROJECT_DIR}/RoboFlamingo/models/mpt-1b-redpajama-200b",
            "vision_encoder_path": "ViT-L-14",
            "vision_encoder_pretrained": "openai",
            "vision_encoder_cache_dir": f"{PROJECT_DIR}/RoboFlamingo/models/clip",
            "cross_attn_every_n_layers": 1,
            "openflamingo_checkpoint": f"{PROJECT_DIR}/RoboFlamingo/models/open_flamingo/checkpoint.pt",
            "decoder_layers_attr_name": "transformer.blocks",
            "window_size":12,  # must be the same as DATASET_CONFIG
            "resampler_type": "depth_codebook",  # 这里的resampler_type不会去修改模型的resampler_type，只是用于区分不同的模型
            "extra_config": {}
        },
        "mpt_3b_depth_depth_codebook_ema": {
            "load_fn": "load_DepthMPTFlamingo", 
            "lang_encoder_path": f"{PROJECT_DIR}/RoboFlamingo/models/mpt-1b-redpajama-200b",
            "vision_encoder_path": "ViT-L-14",
            "vision_encoder_pretrained": "openai",
            "vision_encoder_cache_dir": f"{PROJECT_DIR}/RoboFlamingo/models/clip",
            "cross_attn_every_n_layers": 1,
            "openflamingo_checkpoint": f"{PROJECT_DIR}/RoboFlamingo/models/open_flamingo/checkpoint.pt",
            "decoder_layers_attr_name": "transformer.blocks",
            "window_size":12,  # must be the same as DATASET_CONFIG
            "resampler_type": "depth_codebook_ema",  # 这里的resampler_type不会去修改模型的resampler_type，只是用于区分不同的模型
            "extra_config": {}
        },
        "mpt_3b_depth_depth_codebook_ema_finetune": {
            # 这个模型会加载vq_alpha=1中训练的codebook, 训练过程中冻结rgb encoder, depth encoder, depth codebook, 只训练policy和文本模型
            "load_fn": "load_DepthMPTFlamingo", 
            "lang_encoder_path": f"{PROJECT_DIR}/RoboFlamingo/models/mpt-1b-redpajama-200b",
            "vision_encoder_path": "ViT-L-14",
            "vision_encoder_pretrained": "openai",
            "vision_encoder_cache_dir": f"{PROJECT_DIR}/RoboFlamingo/models/clip",
            "cross_attn_every_n_layers": 1,
            "openflamingo_checkpoint": f"{PROJECT_DIR}/RoboFlamingo/models/open_flamingo/checkpoint.pt",
            "decoder_layers_attr_name": "transformer.blocks",
            "window_size":12,  # must be the same as DATASET_CONFIG
            "resampler_type": "depth_codebook_ema_finetune",  # 这里的resampler_type不会去修改模型的resampler_type，只是用于区分不同的模型
            "extra_config": {}
        },
        "mpt_3b_depth_mm_prompt": {
            "load_fn": "load_DepthMPTFlamingo", 
            "lang_encoder_path": f"{PROJECT_DIR}/RoboFlamingo/models/mpt-1b-redpajama-200b",
            "vision_encoder_path": "ViT-L-14",
            "vision_encoder_pretrained": "openai",
            "vision_encoder_cache_dir": f"{PROJECT_DIR}/RoboFlamingo/models/clip",
            "cross_attn_every_n_layers": 1,
            "openflamingo_checkpoint": f"{PROJECT_DIR}/RoboFlamingo/models/open_flamingo/checkpoint.pt",
            "decoder_layers_attr_name": "transformer.blocks",
            "window_size":12,  # must be the same as DATASET_CONFIG
            "resampler_type": "MM_Prompt",
            "extra_config": {
            }
        },
    }
}

DATASET_CONFIG = {
    "LiberoDataset": {
        "load_fn": "load_LiberoDataset",
        "dataset_dir": "data/LIBERO/v1",
        "window_size": 12,
        "pad": True,
        "rgb_pad": 10,
        "gripper_pad": 4,
        "use_ceph": False,  # whether training in the S cluster
        "used_classes": ["libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial"],
        "metadata_file": "metadata/libero.txt",
    },
    "DepthLiberoDataset": {
        "load_fn": "load_DepthLiberoDataset",
        "dataset_dir": "data/LIBERO/v1",
        "window_size": 12,
        "pad": True,
        "rgb_pad": 10,
        "gripper_pad": 4,
        "use_ceph": True,  # whether training in the S cluster
        "used_classes": ["libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial"],
        "metadata_file": "metadata/libero.txt",
    },
}

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

def get_config(key, config):
    if "-" not in key: 
        return config[key]
    if "-" in key:
        key0, key1 = key.split("-")
        return get_config(key1, config[key0])

def load_dataset(key, image_processor, text_tokenizer):
    dataset_config = get_config(key, DATASET_CONFIG)
    if dataset_config["load_fn"] == "load_LiberoDataset":
        def _preprocess_image(sample, image_processor):
            """
            sample: [<class 'PIL.Image.Image'>, ...]
            Return: Tensor shape = (seq_len, 3, 224, 224)
            """
            image = [image_processor(s).unsqueeze(0) for s in sample]  # [(1, 3, 224, 224), ...]
            image = torch.cat(image, dim=0)
            return image
        def _preprocess_text(sample, text_tokenizer):
            text_tokenizer.padding_side = "right"
            sample = [
                # (f"{s.strip()}{tokenizer.eos_token}")
                # for s in sample
                (f"<image>{s.strip()}<|endofchunk|>{text_tokenizer.eos_token}") for s in sample
            ]
            text = text_tokenizer(
                sample,
                max_length=32,
                padding="longest",
                truncation="only_first",
                return_tensors="pt",
            )
            return text["input_ids"], text["attention_mask"]
        dataset = load_LiberoDataset(
            image_fn=functools.partial(_preprocess_image, image_processor=image_processor),
            text_fn=functools.partial(_preprocess_text, text_tokenizer=text_tokenizer), 
            dataset_dir=dataset_config["dataset_dir"],
            window_size=dataset_config["window_size"],
            pad=dataset_config["pad"],
            rgb_pad=dataset_config["rgb_pad"],
            gripper_pad=dataset_config["gripper_pad"],
            used_classes=dataset_config["used_classes"],
            metadata_file=dataset_config["metadata_file"],
        )
        return dataset
    elif dataset_config["load_fn"] == "load_DepthLiberoDataset":
        def _preprocess_image(sample, image_processor):
            """
            sample: [<class 'PIL.Image.Image'>, ...]
            Return: Tensor shape = (seq_len, 3, 224, 224)
            """
            image = [image_processor(s).unsqueeze(0) for s in sample]  # [(1, 3, 224, 224), ...]
            image = torch.cat(image, dim=0)
            return image
        def _preprocess_text(sample, text_tokenizer):
            text_tokenizer.padding_side = "right"
            sample = [
                # (f"{s.strip()}{tokenizer.eos_token}")
                # for s in sample
                (f"<image>{s.strip()}<|endofchunk|>{text_tokenizer.eos_token}") for s in sample
            ]
            text = text_tokenizer(
                sample,
                max_length=32,
                padding="longest",
                truncation="only_first",
                return_tensors="pt",
            )
            return text["input_ids"], text["attention_mask"]
        dataset = load_DepthLiberoDataset(
            image_fn=functools.partial(_preprocess_image, image_processor=image_processor),
            text_fn=functools.partial(_preprocess_text, text_tokenizer=text_tokenizer), 
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
    else:
        raise NotImplementedError

def load_model(key):
    model_config = get_config(key, MODEL_CONFIG)
    if model_config["load_fn"] == "load_MPTFlamingo":
        model, image_processor, text_tokenizer = load_MPTFlamingo(
            lang_encoder_path=model_config["lang_encoder_path"],
            vision_encoder_path=model_config["vision_encoder_path"],
            vision_encoder_pretrained=model_config["vision_encoder_pretrained"],
            vision_encoder_cache_dir=model_config["vision_encoder_cache_dir"],
            decoder_layers_attr_name=model_config["decoder_layers_attr_name"],
            cross_attn_every_n_layers=model_config["cross_attn_every_n_layers"],
            window_size=model_config["window_size"],
            freeze_lm=model_config.get("freeze_lm", False)
        )
        openflamingo_checkpoint = model_config["openflamingo_checkpoint"]
        model.load_state_dict(torch.load(openflamingo_checkpoint, map_location="cpu"), strict=False)
        extra_config = model_config.get("extra_config", {})
        return model, image_processor, text_tokenizer, extra_config
    elif model_config["load_fn"] == "load_DepthMPTFlamingo":
        model, image_processor, text_tokenizer = load_DepthMPTFlamingo(
            lang_encoder_path=model_config["lang_encoder_path"],
            vision_encoder_path=model_config["vision_encoder_path"],
            vision_encoder_pretrained=model_config["vision_encoder_pretrained"],
            vision_encoder_cache_dir=model_config["vision_encoder_cache_dir"],
            decoder_layers_attr_name=model_config["decoder_layers_attr_name"],
            cross_attn_every_n_layers=model_config["cross_attn_every_n_layers"],
            window_size=model_config["window_size"],
            freeze_lm=model_config.get("freeze_lm", False),
            freeze_rgb=model_config.get("freeze_rgb", False),
            resampler_type=model_config.get("resampler_type", "perceiver_resampler"),
        )
        openflamingo_checkpoint = model_config["openflamingo_checkpoint"]
        model.load_state_dict(torch.load(openflamingo_checkpoint, map_location="cpu"), strict=False)
        extra_config = model_config.get("extra_config", {})
        return model, image_processor, text_tokenizer, extra_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--backend", type=str, default="nccl")

    parser.add_argument("--run_name", type=str, default="runs/dummy")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, default="dummy")
    parser.add_argument("--dataset_name", type=str, default="DummyDataset")
    parser.add_argument("--checkpoint", type=str, default="")
    
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=5000)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--gripper_loss_weight", type=float, default=0.01)

    # deepspeed config
    # parser.add_argument("--train_batch_size", type=int, default=8)  # train_batch_size = train_micro_batch_size_per_gpu * gradient_accumulation_steps * num_gpus
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=4)
    parser.add_argument("--steps_per_print", type=int, default=100)
    parser.add_argument("--wall_clock_breakdown", type=bool, default=False)

    parser.add_argument("--auto_remove_prev_ckpt", action="store_true")

    parser.add_argument(
        "--train_type", 
        type=str, 
        default="normal", 
        choices=[
            "normal",
            "missing_modality", 
            "CMKD",
            "depth_codebook_ema_finetune", 
        ]
    )
    
    parser.add_argument("--codebook_loss_weight", type=float, default=1.0)  # only used in depth_codebook
    parser.add_argument("--codebook_train_step", type=int, default=500)

    parser.add_argument("--cmkd_loss_weight", type=float, default=1.0)  # only used in CMKD

    parser.add_argument("--record_grad_norm", action="store_true")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

def train():
    args = parse_args()
    if args.train_type == "missing_modality":
        np.random.seed(args.seed)
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
                if args.record_grad_norm:
                    logger_grad = get_logger(
                        filename=f"{args.run_name}/grad-{log_idx}.log",
                        logger_name="roboflamingo-grad",
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
            if args.local_rank == 0 and args.record_grad_norm:
                logger_grad = get_logger(
                    filename=f"{args.run_name}/grad.log",
                    logger_name="roboflamingo-grad",
                )
        print(args)
        with open(f"{args.run_name}/cmd.txt", "w") as f:
            f.write("python " + " ".join(sys.argv) + "\n")
            f.write(json.dumps(vars(args), indent=4) + "\n")

    model, image_processor, text_tokenizer, extra_config = load_model(args.model_name)
    logger.info(f"{model.state_dict().keys()}")
    if args.checkpoint != "" and args.checkpoint != "none":
        args.checkpoint = args.checkpoint.split("#")
        for checkpoint in args.checkpoint:
            logger.info(f"load checkpoint from {checkpoint}")
            ckpt = torch.load(checkpoint, map_location="cpu")
            state_dict = OrderedDict([(k.replace("module.", ""), v) for k, v in ckpt["model_state_dict"].items()])
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")
            logger.info(f"{checkpoint} missing_keys: {missing_keys}")
            logger.info(f"{checkpoint} unexpected_keys: {unexpected_keys}")
    train_dataset = load_dataset(args.dataset_name, image_processor, text_tokenizer)
    
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
    config.update(extra_config)
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
    
    for epoch in range(resume_epoch, args.num_epochs):
        t = tqdm(
            enumerate(dataloader),
            disable=engine.local_rank != 0,
            total=len(dataloader),
        )
        t.set_description(f"epoch {epoch+1}/{args.num_epochs}")
        t1 = time.time()
        time_avgmeter = common_utils.AverageMeter()
        for num_steps, batch in t:
            if type(batch) == list or type(batch) == tuple:
                (vision_x, vision_gripper), actions, (lang_x, attention_mask) = batch
                vision_x = (vision_x.to(engine.local_rank, non_blocking=True).unsqueeze(2).unsqueeze(2))
                vision_gripper = (vision_gripper.to(engine.local_rank, non_blocking=True).unsqueeze(2).unsqueeze(2))  # (B, window_size, 1, 1, 3, 224, 224)

                lang_x = lang_x.to(engine.local_rank, non_blocking=True).unsqueeze(1).repeat(1, vision_x.shape[1], 1)  # (B, window_size, 13)
                attention_mask = attention_mask.to(engine.local_rank, non_blocking=True).unsqueeze(1).repeat(1, vision_x.shape[1], 1)  # (B, window_size, 13)

                actions = actions.to(engine.local_rank, non_blocking=True)  # (B, window_size, 7)
                actions = [actions[..., :6], (actions[..., 6:] + 1) // 2]
                output = engine(vision_x, lang_x, attention_mask, vision_gripper)
            else:
                assert type(batch) == dict
                vision_x = batch["image_tensors"]
                vision_gripper = batch["gripper_tensors"]
                actions = batch["action_tensors"]
                lang_x = batch["text_tensors"]
                attention_mask = batch["attention_mask"]
                depth_static = batch["depth_static_tensors"]
                depth_gripper = batch["depth_gripper_tensors"]

                vision_x = (vision_x.to(engine.local_rank, non_blocking=True).unsqueeze(2).unsqueeze(2))
                vision_gripper = (vision_gripper.to(engine.local_rank, non_blocking=True).unsqueeze(2).unsqueeze(2))  # (B, window_size, 1, 1, 3, 224, 224)
                depth_static = (depth_static.to(engine.local_rank, non_blocking=True).unsqueeze(2).unsqueeze(2))
                depth_gripper = (depth_gripper.to(engine.local_rank, non_blocking=True).unsqueeze(2).unsqueeze(2))  # (B, window_size, 1, 1, 3, 224, 224)

                lang_x = lang_x.to(engine.local_rank, non_blocking=True).unsqueeze(1).repeat(1, vision_x.shape[1], 1)  # (B, window_size, 13)
                attention_mask = attention_mask.to(engine.local_rank, non_blocking=True).unsqueeze(1).repeat(1, vision_x.shape[1], 1)  # (B, window_size, 13)

                actions = actions.to(engine.local_rank, non_blocking=True)  # (B, window_size, 7)
                actions = [actions[..., :6], (actions[..., 6:] + 1) // 2]
                if args.train_type == "missing_modality":
                    t = np.random.uniform(0, 1)
                    if t < 0.3:
                        output = engine(vision_x, lang_x, attention_mask, vision_gripper, None, None)
                    elif t < 0.6:
                        output = engine(None, lang_x, attention_mask, None, depth_static, depth_gripper)
                    else:
                        output = engine(vision_x, lang_x, attention_mask, vision_gripper, depth_static, depth_gripper)
                elif args.train_type == "depth_codebook_ema_finetune":
                    assert args.model_name == "roboflamingo-mpt_3b_depth_depth_codebook_ema_finetune"
                    output = engine(vision_x, lang_x, attention_mask, vision_gripper, depth_static, depth_gripper, mode="inference")
                else:
                    output = engine(vision_x, lang_x, attention_mask, vision_gripper, depth_static, depth_gripper, mode="training")

             # compute loss

            if args.train_type == "normal" or args.train_type == "missing_modality":
                num_actions, bin_actions = output.logits[0], output.logits[1]  # (B, window_size, 6), (B, window_size, 1)
                loss_num = torch.nn.functional.huber_loss(num_actions, actions[0])
                loss_bin = torch.nn.functional.binary_cross_entropy(bin_actions, actions[1])
                total_loss = loss_num + loss_bin * args.gripper_loss_weight
                engine.backward(total_loss)
                if num_steps % engine.gradient_accumulation_steps() == 0 and args.record_grad_norm and step % 10 == 0 and engine.local_rank == 0:
                    assert args.model_name == "roboflamingo-mpt_3b_depth"
                    from deepspeed.utils import safe_get_full_grad
                    grad_perceiver = safe_get_full_grad(model.perceiver.norm.weight)
                    grad_depth_perceiver = safe_get_full_grad(model.depth_perceiver.norm.weight)
                    grad_perceiver_norm = torch.norm(grad_perceiver).detach().cpu().numpy().item()
                    grad_depth_perceiver_norm = torch.norm(grad_depth_perceiver).detach().cpu().numpy().item()
                    logger_grad.info(f"step: {step:3d} grad_perceiver_norm: {grad_perceiver_norm} grad_depth_perceiver_norm: {grad_depth_perceiver_norm}")
                engine.step()

                if num_steps % engine.gradient_accumulation_steps() == 0:
                    step += 1
                    if step % args.steps_per_print == 0:
                        print(f'local rank: {engine.local_rank} step: {step:3d} loss: {total_loss}')
                    if step % 10 == 0 and engine.local_rank == 0:
                        time_avgmeter.update(time.time() - t1)
                        t1 = time.time()
                        logger.info(f"step: {step:3d} loss: {total_loss} time: {time_avgmeter.avg}")
            elif args.train_type == "CMKD":
                assert type(batch) != list and type(batch) != tuple, "batch must be a dict(DepthLiberoDataset)"
                num_actions, bin_actions = output.logits[0], output.logits[1]  # (B, window_size, 6), (B, window_size, 1)
                vision_x, vision_gripper = output.vision_x, output.vision_gripper  # (B, window_size, 1, num_latents, vision_embedding_dim)
                depth_static, depth_gripper = output.depth_static, output.depth_gripper  # (B, window_size, 1, num_latents, vision_embedding_dim)
                loss_num = torch.nn.functional.huber_loss(num_actions, actions[0])
                loss_bin = torch.nn.functional.binary_cross_entropy(bin_actions, actions[1])
                action_loss = loss_num + loss_bin * args.gripper_loss_weight
                cmkd_loss_num = torch.nn.functional.mse_loss(vision_x, depth_static)
                cmkd_loss_bin = torch.nn.functional.mse_loss(vision_gripper, depth_gripper)
                cmkd_loss = cmkd_loss_num + cmkd_loss_bin
                total_loss = action_loss + cmkd_loss * args.cmkd_loss_weight
                engine.backward(total_loss)
                engine.step()

                if num_steps % engine.gradient_accumulation_steps() == 0:
                    step += 1
                    if step % args.steps_per_print == 0:
                        print(f'local rank: {engine.local_rank} step: {step:3d} loss: {total_loss} action_loss: {action_loss} cmkd_loss: {cmkd_loss}')
                    if step % 10 == 0 and engine.local_rank == 0:
                        time_avgmeter.update(time.time() - t1)
                        t1 = time.time()
                        logger.info(f"step: {step:3d} loss: {total_loss} action_loss: {action_loss} cmkd_loss: {cmkd_loss} time: {time_avgmeter.avg}")
            elif args.train_type == "pred_depth":
                assert args.model_name == "roboflamingo-mpt_3b_depth_pred_depth"
                num_actions, bin_actions = output.logits[0], output.logits[1]  # (B, window_size, 6), (B, window_size, 1)
                loss_num = torch.nn.functional.huber_loss(num_actions, actions[0])
                loss_bin = torch.nn.functional.binary_cross_entropy(bin_actions, actions[1])
                loss_pred_depth = output.depth_pred_loss
                total_loss = loss_num + loss_bin * args.gripper_loss_weight + loss_pred_depth
                engine.backward(total_loss)
                engine.step()

                if num_steps % engine.gradient_accumulation_steps() == 0:
                    step += 1
                    if step % args.steps_per_print == 0:
                        print(f'local rank: {engine.local_rank} step: {step:3d} loss: {total_loss} loss_pred_depth: {loss_pred_depth}')
                    if step % 10 == 0 and engine.local_rank == 0:
                        time_avgmeter.update(time.time() - t1)
                        t1 = time.time()
                        logger.info(f"step: {step:3d} loss: {total_loss} loss_pred_depth: {loss_pred_depth} time: {time_avgmeter.avg}")
            elif args.train_type == "depth_codebook_ema_finetune":
                assert args.model_name == "roboflamingo-mpt_3b_depth_depth_codebook_ema_finetune"
                num_actions, bin_actions = output.logits[0], output.logits[1]  # (B, window_size, 6), (B, window_size, 1)
                loss_num = torch.nn.functional.huber_loss(num_actions, actions[0])
                loss_bin = torch.nn.functional.binary_cross_entropy(bin_actions, actions[1])
                codebook_loss_static = output.loss_static
                codebook_perplexity_static = output.perplexity_static
                codebook_loss_gripper = output.loss_gripper
                codebook_perplexity_gripper = output.perplexity_gripper
                indices_static = output.indices_static
                total_loss = loss_num + loss_bin * args.gripper_loss_weight
                engine.backward(total_loss)
                engine.step()

                if num_steps % engine.gradient_accumulation_steps() == 0:
                    step += 1
                    if step % args.steps_per_print == 0:
                        print(f'local rank: {engine.local_rank} step: {step:3d} loss: {total_loss} codebook_loss_static: {codebook_loss_static} codebook_loss_gripper: {codebook_loss_gripper} codebook_perplexity_static: {codebook_perplexity_static} codebook_perplexity_gripper: {codebook_perplexity_gripper} action_loss: {loss_num + loss_bin * args.gripper_loss_weight} indices_static: {indices_static.sum()}')
                    if step % 10 == 0 and engine.local_rank == 0:
                        time_avgmeter.update(time.time() - t1)
                        t1 = time.time()
                        logger.info(f"step: {step:3d} loss: {total_loss} codebook_loss_static: {codebook_loss_static} codebook_loss_gripper: {codebook_loss_gripper} codebook_perplexity_static: {codebook_perplexity_static} codebook_perplexity_gripper: {codebook_perplexity_gripper} action_loss: {loss_num + loss_bin * args.gripper_loss_weight} indices_static: {indices_static.sum()} time: {time_avgmeter.avg}")
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
    train()
