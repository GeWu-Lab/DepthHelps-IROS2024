import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ["MUJOCO_GL"] = "osmesa"
import sys
import json
import random
import argparse
import functools
from tqdm import tqdm
from collections import deque, OrderedDict

import cv2
import torch
import imageio
import numpy as np
np.int = np.int64
np.float = np.float32
from PIL import Image
from torchvision import transforms

from libero.libero.utils.time_utils import Timer
from libero.libero.utils.video_utils import VideoWriter
from libero.libero.benchmark import get_benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv

from models.roboflamingo.flamingo_mpt import load_MPTFlamingo
from models.roboflamingo.flamingo_mpt_depth import load_DepthMPTFlamingo
from datasets.libero_dataset import load_LiberoDataset
from datasets.depth_libero_dataset import DepthNorm, OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

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
            "model_type": "pytorch",
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
        "mpt_3b_depth_missing_modality": {
            "load_fn": "load_DepthMPTFlamingo_pred_depth", 
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
        "mpt_3b_depth_mm_prompt": {
            "load_fn": "load_DepthMPTFlamingo_pred_depth", 
            "lang_encoder_path": f"{PROJECT_DIR}/RoboFlamingo/models/mpt-1b-redpajama-200b",
            "vision_encoder_path": "ViT-L-14",
            "vision_encoder_pretrained": "openai",
            "vision_encoder_cache_dir": f"{PROJECT_DIR}/RoboFlamingo/models/clip",
            "cross_attn_every_n_layers": 1,
            "openflamingo_checkpoint": f"{PROJECT_DIR}/RoboFlamingo/models/open_flamingo/checkpoint.pt",
            "decoder_layers_attr_name": "transformer.blocks",
            "window_size":12,  # must be the same as DATASET_CONFIG
            "resampler_type": "MM_Prompt",
            "set_depth_to_none": True,
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
            "extra_config": {
            }
        },
        "mpt_3b_depth_depth_codebook_ema_finetune": {
            "load_fn": "load_DepthMPTFlamingo_pred_depth", 
            "lang_encoder_path": f"{PROJECT_DIR}/RoboFlamingo/models/mpt-1b-redpajama-200b",
            "vision_encoder_path": "ViT-L-14",
            "vision_encoder_pretrained": "openai",
            "vision_encoder_cache_dir": f"{PROJECT_DIR}/RoboFlamingo/models/clip",
            "cross_attn_every_n_layers": 1,
            "openflamingo_checkpoint": f"{PROJECT_DIR}/RoboFlamingo/models/open_flamingo/checkpoint.pt",
            "decoder_layers_attr_name": "transformer.blocks",
            "window_size":12,  # must be the same as DATASET_CONFIG
            "resampler_type": "depth_codebook_ema_pred_depth",
            "extra_config": {},
        },
    },
}
BENCHMARKS = ["LIBERO_OBJECT", "LIBERO_10", "LIBERO_SPATIAL", "LIBERO_GOAL"]

def depthimg2Meters(depth, near=0.01060981611847304, far=530.490780726692):
    image = near / (1 - depth * (1 - near / far))
    return image

class VideoWriter2(VideoWriter):

    def __init__(self, start_idx, video_path, save_video=False, fps=30, single_video=True):
        self.start_idx = start_idx
        super().__init__(video_path, save_video, fps, single_video)

    def save(self):
        if self.save_video:
            os.makedirs(self.video_path, exist_ok=True)
            if self.single_video:
                video_name = os.path.join(self.video_path, f"video.mp4")
                video_writer = imageio.get_writer(video_name, fps=self.fps)
                for idx in self.image_buffer.keys():
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im)
                video_writer.close()
            else:
                for idx in self.image_buffer.keys():
                    video_name = os.path.join(self.video_path, f"{self.start_idx + idx}.mp4")
                    video_writer = imageio.get_writer(video_name, fps=self.fps)
                    for im in self.image_buffer[idx]:
                        video_writer.append_data(im)
                    video_writer.close()
            print(f"Saved videos to {self.video_path}.")

def get_config(key, config):
    if "-" not in key: 
        return config[key]
    if "-" in key:
        key0, key1 = key.split("-")
        return get_config(key1, config[key0])

def create_model(model_name: str, checkpoint_type: str, checkpoint: str):
    model_config = get_config(model_name, MODEL_CONFIG)
    if model_config["load_fn"] == "load_MPTFlamingo":
        model, image_processor, text_tokenizer = load_MPTFlamingo(
            lang_encoder_path=model_config["lang_encoder_path"],
            vision_encoder_path=model_config["vision_encoder_path"],
            vision_encoder_pretrained=model_config["vision_encoder_pretrained"],
            vision_encoder_cache_dir=model_config["vision_encoder_cache_dir"],
            decoder_layers_attr_name=model_config["decoder_layers_attr_name"],
            cross_attn_every_n_layers=model_config["cross_attn_every_n_layers"],
            window_size=model_config["window_size"],
        )
        openflamingo_checkpoint = model_config["openflamingo_checkpoint"]
        model.load_state_dict(torch.load(openflamingo_checkpoint, map_location="cpu"), strict=False)
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
        class ModelWrapper:

            def __init__(self, model, tokenizer, image_processor, device = 'cuda', future_act_len=-1) -> None:
                self.model = model
                self.model.to(device)
                self.model.eval()
                self.tokenizer = tokenizer
                self.image_processor = image_processor
                self.device = device
                self.history_len = self.model.window_size
                self.future_act_len = future_act_len
                self.text_process_fn = functools.partial(_preprocess_text, text_tokenizer=self.tokenizer)
                self.image_process_fn = functools.partial(_preprocess_image, image_processor=self.image_processor)
                self.action_hist_queue = []
                self.feature_cache = None
                self.dt_feat_cache = []
                self.img_queue = deque(maxlen=self.history_len)
                self.gripper_queue = deque(maxlen=self.history_len)
                self.state_queue = deque(maxlen=self.history_len)
                self.mask_queue = deque(maxlen=self.history_len)
                self.text_queue = deque(maxlen=self.history_len)

            def reset(self):
                self.feature_cache = None
                self.dt_feat_cache = []
                self.img_queue = deque(maxlen=self.history_len)
                self.gripper_queue = deque(maxlen=self.history_len)
                self.state_queue = deque(maxlen=self.history_len)
                self.mask_queue = deque(maxlen=self.history_len)
                self.text_queue = deque(maxlen=self.history_len)
                self.model.lm_head.hidden_state = None
                self.model.lm_head.history_memory = []

            def step(self, obs, goal, get_action=True):
                
                # expand image dimension
                image = obs["rgb_obs"]["rgb_static"]
                image = Image.fromarray(image)
                image_x = self.image_process_fn([image])  # (1, 3, 224, 224)
                image_x = image_x.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(dtype=torch.float32)  # (1, 1, 1, 1, 3, 224, 224)
                
                # expand text dimension
                lang_x, attention_mask = self.text_process_fn([goal])  # (1, 13), (1, 13)
                lang_x = lang_x.unsqueeze(1)  # (1, 1, 13)
                attention_mask = attention_mask.unsqueeze(1)  # (1, 1, 13)
                
                vision_gripper = obs["rgb_obs"]['rgb_gripper']
                vision_gripper = Image.fromarray(vision_gripper)
                vision_gripper = self.image_process_fn([vision_gripper])  # (1, 3, 224, 224)
                vision_gripper = vision_gripper.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(dtype=torch.float32)  # (1, 1, 1, 1, 3, 224, 224)

                window_size = self.model.lm_head.window_size
                self.model.lm_head.window_size = 1
                with torch.no_grad():
                    image_x = image_x.to(self.device)
                    lang_x = lang_x.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    vision_gripper = vision_gripper.to(self.device)

                    action = self.model(
                        vision_x=image_x,
                        lang_x=lang_x,
                        attention_mask=attention_mask,
                        vision_gripper=vision_gripper,
                    )  # (1, 1, 6), (1, 1, 1)
                    action = torch.concat((action.logits[0], action.logits[1] > 0.5), dim=2).squeeze(0)[-1]  # (1, 1, 7) -> (7,)
                    action[-1] = 2 * action[-1] - 1
                    action = action.cpu().detach().to(dtype=torch.float16).numpy()
                self.model.lm_head.window_size = window_size
                return action

            def _preprocess_image(self, sample):
                """
                sample: [<class 'PIL.Image.Image'>, ...]
                Return: Tensor shape = (seq_len, 3, 224, 224)
                """
                image = [self.image_processor(s).unsqueeze(0) for s in sample]  # [(1, 3, 224, 224), ...]
                image = torch.cat(image, dim=0)
                return image

            def _preprocess_text(self, sample):
                self.tokenizer.padding_side = "right"
                sample = [
                    # (f"{s.strip()}{tokenizer.eos_token}")
                    # for s in sample
                    (f"<image>{s.strip()}<|endofchunk|>{self.tokenizer.eos_token}") for s in sample
                ]
                text = text_tokenizer(
                    sample,
                    max_length=32,
                    padding="longest",
                    truncation="only_first",
                    return_tensors="pt",
                )
                return text["input_ids"], text["attention_mask"]

            def step_batch(self, obs, goals, get_action=True):
                imgs_static = []
                imgs_gripper = []

                # expand image dimension
                for i in range(len(obs)):
                    image = obs[i]["rgb_obs"]["rgb_static"]
                    image = Image.fromarray(image)
                    imgs_static.append(image)

                    vision_gripper = obs[i]["rgb_obs"]['rgb_gripper']
                    vision_gripper = Image.fromarray(vision_gripper)
                    imgs_gripper.append(vision_gripper)

                imgs_static = self._preprocess_image(imgs_static)  # (B, 3, 224, 224)
                imgs_gripper = self._preprocess_image(imgs_gripper)  # (B, 3, 224, 224)
                lang_x, attention_mask = self._preprocess_text(goals)  # (B, T_text), (B, T_text)

                imgs_static = imgs_static.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(dtype=torch.float32)  # (B, 1, 1, 1, 3, 224, 224)
                imgs_gripper = imgs_gripper.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(dtype=torch.float32)  # (B, 1, 1, 1, 3, 224, 224)
                lang_x = lang_x.unsqueeze(1)  # (B, 1, T_text)
                attention_mask = attention_mask.unsqueeze(1)  # (B, 1, T_text)

                window_size = self.model.lm_head.window_size
                self.model.lm_head.window_size = 1

                with torch.no_grad():
                    imgs_static = imgs_static.to(self.device)
                    lang_x = lang_x.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    imgs_gripper = imgs_gripper.to(self.device)

                    action = self.model(
                        vision_x=imgs_static,
                        lang_x=lang_x,
                        attention_mask=attention_mask,
                        vision_gripper=imgs_gripper,
                    )  # (B, 1, 6), (B, 1, 1)
                    action = torch.concat((action.logits[0], action.logits[1] > 0.5), dim=2)[:, 0]  # (B, 1, 7) -> (B, 7)
                    action[:, -1] = 2 * action[:, -1] - 1
                    action = action.cpu().detach().to(dtype=torch.float16).numpy()
                self.model.lm_head.window_size = window_size
                return action

        if checkpoint_type == "pytorch":
            ckpt = torch.load(checkpoint, map_location="cpu")
            state_dict = OrderedDict([(k.replace("module.", ""), v) for k, v in ckpt["model_state_dict"].items()])
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")
        elif checkpoint_type == "deepspeed":
            ckpt = torch.load(checkpoint, map_location="cpu")
            state_dict = ckpt["module"]
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")
        model_wrapper = ModelWrapper(model, text_tokenizer, image_processor)
        return model_wrapper
    elif model_config["load_fn"] == "load_DepthMPTFlamingo":
        model, image_processor, text_tokenizer = load_DepthMPTFlamingo(
            lang_encoder_path=model_config["lang_encoder_path"],
            vision_encoder_path=model_config["vision_encoder_path"],
            vision_encoder_pretrained=model_config["vision_encoder_pretrained"],
            vision_encoder_cache_dir=model_config["vision_encoder_cache_dir"],
            decoder_layers_attr_name=model_config["decoder_layers_attr_name"],
            cross_attn_every_n_layers=model_config["cross_attn_every_n_layers"],
            window_size=model_config["window_size"],
            resampler_type=model_config.get("resampler_type", "perceiver_resampler"),
        )
        openflamingo_checkpoint = model_config["openflamingo_checkpoint"]
        model.load_state_dict(torch.load(openflamingo_checkpoint, map_location="cpu"), strict=False)
        
        class ModelWrapper:

            def __init__(self, model, tokenizer, image_processor, device = 'cuda', future_act_len=-1) -> None:
                self.model = model
                self.model.to(device)
                self.model.eval()
                self.tokenizer = tokenizer
                self.image_processor = image_processor
                self.device = device
                self.history_len = self.model.window_size
                self.future_act_len = future_act_len
                self.action_hist_queue = []
                self.feature_cache = None
                self.dt_feat_cache = []
                self.img_queue = deque(maxlen=self.history_len)
                self.gripper_queue = deque(maxlen=self.history_len)
                self.depth_static_queue = deque(maxlen=self.history_len)
                self.depth_gripper_queue = deque(maxlen=self.history_len)
                self.state_queue = deque(maxlen=self.history_len)
                self.mask_queue = deque(maxlen=self.history_len)
                self.text_queue = deque(maxlen=self.history_len)
                self.depth_transform = transforms.Compose(
                    [
                        DepthNorm(max_depth=5.0),
                        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(224),
                        transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),  # assume image
                    ]
                )

            def reset(self):
                self.feature_cache = None
                self.dt_feat_cache = []
                self.img_queue = deque(maxlen=self.history_len)
                self.gripper_queue = deque(maxlen=self.history_len)
                self.depth_static_queue = deque(maxlen=self.history_len)
                self.depth_gripper_queue = deque(maxlen=self.history_len)
                self.state_queue = deque(maxlen=self.history_len)
                self.mask_queue = deque(maxlen=self.history_len)
                self.text_queue = deque(maxlen=self.history_len)
                self.model.lm_head.hidden_state = None
                self.model.lm_head.history_memory = []

            def _preprocess_image(self, sample):
                """
                sample: [<class 'PIL.Image.Image'>, ...]
                Return: Tensor shape = (seq_len, 3, 224, 224)
                """
                image = [self.image_processor(s).unsqueeze(0) for s in sample]  # [(1, 3, 224, 224), ...]
                image = torch.cat(image, dim=0)
                return image

            def _preprocess_text(self, sample):
                self.tokenizer.padding_side = "right"
                sample = [
                    # (f"{s.strip()}{tokenizer.eos_token}")
                    # for s in sample
                    (f"<image>{s.strip()}<|endofchunk|>{self.tokenizer.eos_token}") for s in sample
                ]
                text = text_tokenizer(
                    sample,
                    max_length=32,
                    padding="longest",
                    truncation="only_first",
                    return_tensors="pt",
                )
                return text["input_ids"], text["attention_mask"]
            
            def _depth_fn(self, sample):
                image = [self.depth_transform(s).unsqueeze(0) for s in sample]  # [(1, 3, 224, 224), ...]
                image = torch.cat(image, dim=0)
                return image

            def step_batch(self, obs, goals, get_action=True):
                imgs_static = []
                imgs_gripper = []
                depth_imgs_static = []
                depth_imgs_gripper = []

                # expand image dimension
                for i in range(len(obs)):
                    image = obs[i]["rgb_obs"]["rgb_static"]
                    image = Image.fromarray(image)
                    imgs_static.append(image)

                    vision_gripper = obs[i]["rgb_obs"]['rgb_gripper']
                    vision_gripper = Image.fromarray(vision_gripper)
                    imgs_gripper.append(vision_gripper)

                    depth_imgs_static.append(obs[i]["depth_obs"]["depth_static"][:, :, 0])
                    depth_imgs_gripper.append(obs[i]["depth_obs"]["depth_gripper"][:, :, 0])

                imgs_static = self._preprocess_image(imgs_static)  # (B, 3, 224, 224)
                imgs_gripper = self._preprocess_image(imgs_gripper)  # (B, 3, 224, 224)
                depth_imgs_static = self._depth_fn(depth_imgs_static)  # (B, 3, 224, 224)
                depth_imgs_gripper = self._depth_fn(depth_imgs_gripper)  # (B, 3, 224, 224)
                lang_x, attention_mask = self._preprocess_text(goals)  # (B, T_text), (B, T_text)

                imgs_static = imgs_static.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(dtype=torch.float32)  # (B, 1, 1, 1, 3, 224, 224)
                imgs_gripper = imgs_gripper.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(dtype=torch.float32)  # (B, 1, 1, 1, 3, 224, 224)
                depth_imgs_static = depth_imgs_static.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(dtype=torch.float32)  # (B, 1, 1, 1, 3, 224, 224)
                depth_imgs_gripper = depth_imgs_gripper.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(dtype=torch.float32)  # (B, 1, 1, 1, 3, 224, 224)
                lang_x = lang_x.unsqueeze(1)  # (B, 1, T_text)
                attention_mask = attention_mask.unsqueeze(1)  # (B, 1, T_text)

                window_size = self.model.lm_head.window_size
                self.model.lm_head.window_size = 1

                with torch.no_grad():
                    imgs_static = imgs_static.to(self.device)
                    lang_x = lang_x.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    imgs_gripper = imgs_gripper.to(self.device)
                    depth_imgs_static = depth_imgs_static.to(self.device)
                    depth_imgs_gripper = depth_imgs_gripper.to(self.device)
                    depth_imgs_gripper = None if model_config.get("set_depth_to_none", False) else depth_imgs_gripper
                    depth_imgs_static = None if model_config.get("set_depth_to_none", False) else depth_imgs_static
                    action = self.model(
                        vision_x=imgs_static,
                        lang_x=lang_x,
                        attention_mask=attention_mask,
                        vision_gripper=imgs_gripper,
                        depth_static=depth_imgs_static,
                        depth_gripper=depth_imgs_gripper,
                        mode=model_config.get("mode", "training")
                    )  # (B, 1, 6), (B, 1, 1)
                    action = torch.concat((action.logits[0], action.logits[1] > 0.5), dim=2)[:, 0]  # (B, 1, 7) -> (B, 7)
                    action[:, -1] = 2 * action[:, -1] - 1
                    action = action.cpu().detach().to(dtype=torch.float16).numpy()
                self.model.lm_head.window_size = window_size
                return action
        
        if checkpoint_type == "pytorch":
            ckpt = torch.load(checkpoint, map_location="cpu")
            state_dict = OrderedDict([(k.replace("module.", ""), v) for k, v in ckpt["model_state_dict"].items()])
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")
        elif checkpoint_type == "deepspeed":
            ckpt = torch.load(checkpoint, map_location="cpu")
            state_dict = ckpt["module"]
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")
        model_wrapper = ModelWrapper(model, text_tokenizer, image_processor)
        return model_wrapper
    elif model_config["load_fn"] == "load_DepthMPTFlamingo_pred_depth":
        model, image_processor, text_tokenizer = load_DepthMPTFlamingo(
            lang_encoder_path=model_config["lang_encoder_path"],
            vision_encoder_path=model_config["vision_encoder_path"],
            vision_encoder_pretrained=model_config["vision_encoder_pretrained"],
            vision_encoder_cache_dir=model_config["vision_encoder_cache_dir"],
            decoder_layers_attr_name=model_config["decoder_layers_attr_name"],
            cross_attn_every_n_layers=model_config["cross_attn_every_n_layers"],
            window_size=model_config["window_size"],
            resampler_type=model_config.get("resampler_type", "perceiver_resampler"),
        )
        openflamingo_checkpoint = model_config["openflamingo_checkpoint"]
        model.load_state_dict(torch.load(openflamingo_checkpoint, map_location="cpu"), strict=False)
        
        class ModelWrapper:

            def __init__(self, model, tokenizer, image_processor, device = 'cuda', future_act_len=-1) -> None:
                self.model = model
                self.model.to(device)
                self.model.eval()
                self.tokenizer = tokenizer
                self.image_processor = image_processor
                self.device = device
                self.history_len = self.model.window_size
                self.future_act_len = future_act_len
                self.action_hist_queue = []
                self.feature_cache = None
                self.dt_feat_cache = []
                self.img_queue = deque(maxlen=self.history_len)
                self.gripper_queue = deque(maxlen=self.history_len)
                self.depth_static_queue = deque(maxlen=self.history_len)
                self.depth_gripper_queue = deque(maxlen=self.history_len)
                self.state_queue = deque(maxlen=self.history_len)
                self.mask_queue = deque(maxlen=self.history_len)
                self.text_queue = deque(maxlen=self.history_len)
                self.depth_transform = transforms.Compose(
                    [
                        DepthNorm(max_depth=5.0),
                        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                        transforms.CenterCrop(224),
                        transforms.Normalize(OPENAI_DATASET_MEAN, OPENAI_DATASET_STD),  # assume image
                    ]
                )

            def reset(self):
                self.feature_cache = None
                self.dt_feat_cache = []
                self.img_queue = deque(maxlen=self.history_len)
                self.gripper_queue = deque(maxlen=self.history_len)
                self.depth_static_queue = deque(maxlen=self.history_len)
                self.depth_gripper_queue = deque(maxlen=self.history_len)
                self.state_queue = deque(maxlen=self.history_len)
                self.mask_queue = deque(maxlen=self.history_len)
                self.text_queue = deque(maxlen=self.history_len)
                self.model.lm_head.hidden_state = None
                self.model.lm_head.history_memory = []

            def _preprocess_image(self, sample):
                """
                sample: [<class 'PIL.Image.Image'>, ...]
                Return: Tensor shape = (seq_len, 3, 224, 224)
                """
                image = [self.image_processor(s).unsqueeze(0) for s in sample]  # [(1, 3, 224, 224), ...]
                image = torch.cat(image, dim=0)
                return image

            def _preprocess_text(self, sample):
                self.tokenizer.padding_side = "right"
                sample = [
                    # (f"{s.strip()}{tokenizer.eos_token}")
                    # for s in sample
                    (f"<image>{s.strip()}<|endofchunk|>{self.tokenizer.eos_token}") for s in sample
                ]
                text = text_tokenizer(
                    sample,
                    max_length=32,
                    padding="longest",
                    truncation="only_first",
                    return_tensors="pt",
                )
                return text["input_ids"], text["attention_mask"]
            
            def _depth_fn(self, sample):
                image = [self.depth_transform(s).unsqueeze(0) for s in sample]  # [(1, 3, 224, 224), ...]
                image = torch.cat(image, dim=0)
                return image

            def step_batch(self, obs, goals, get_action=True):
                imgs_static = []
                imgs_gripper = []
                depth_imgs_static = []
                depth_imgs_gripper = []

                # expand image dimension
                for i in range(len(obs)):
                    image = obs[i]["rgb_obs"]["rgb_static"]
                    image = Image.fromarray(image)
                    imgs_static.append(image)

                    vision_gripper = obs[i]["rgb_obs"]['rgb_gripper']
                    vision_gripper = Image.fromarray(vision_gripper)
                    imgs_gripper.append(vision_gripper)

                    depth_imgs_static.append(obs[i]["depth_obs"]["depth_static"][:, :, 0])
                    depth_imgs_gripper.append(obs[i]["depth_obs"]["depth_gripper"][:, :, 0])

                imgs_static = self._preprocess_image(imgs_static)  # (B, 3, 224, 224)
                imgs_gripper = self._preprocess_image(imgs_gripper)  # (B, 3, 224, 224)
                depth_imgs_static = self._depth_fn(depth_imgs_static)  # (B, 3, 224, 224)
                depth_imgs_gripper = self._depth_fn(depth_imgs_gripper)  # (B, 3, 224, 224)
                lang_x, attention_mask = self._preprocess_text(goals)  # (B, T_text), (B, T_text)

                imgs_static = imgs_static.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(dtype=torch.float32)  # (B, 1, 1, 1, 3, 224, 224)
                imgs_gripper = imgs_gripper.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(dtype=torch.float32)  # (B, 1, 1, 1, 3, 224, 224)
                depth_imgs_static = depth_imgs_static.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(dtype=torch.float32)  # (B, 1, 1, 1, 3, 224, 224)
                depth_imgs_gripper = depth_imgs_gripper.unsqueeze(1).unsqueeze(1).unsqueeze(1).to(dtype=torch.float32)  # (B, 1, 1, 1, 3, 224, 224)
                lang_x = lang_x.unsqueeze(1)  # (B, 1, T_text)
                attention_mask = attention_mask.unsqueeze(1)  # (B, 1, T_text)

                window_size = self.model.lm_head.window_size
                self.model.lm_head.window_size = 1

                with torch.no_grad():
                    imgs_static = imgs_static.to(self.device)
                    lang_x = lang_x.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    imgs_gripper = imgs_gripper.to(self.device)
                    depth_imgs_static = depth_imgs_static.to(self.device)
                    depth_imgs_gripper = depth_imgs_gripper.to(self.device)

                    action = self.model(
                        vision_x=imgs_static,
                        lang_x=lang_x,
                        attention_mask=attention_mask,
                        vision_gripper=imgs_gripper,
                        # depth_static=depth_imgs_static,
                        # depth_gripper=depth_imgs_gripper,
                        depth_static=None,
                        depth_gripper=None,
                        mode="inference"
                    )  # (B, 1, 6), (B, 1, 1)
                    action = torch.concat((action.logits[0], action.logits[1] > 0.5), dim=2)[:, 0]  # (B, 1, 7) -> (B, 7)
                    action[:, -1] = 2 * action[:, -1] - 1
                    action = action.cpu().detach().to(dtype=torch.float16).numpy()
                self.model.lm_head.window_size = window_size
                return action
        
        if checkpoint_type == "pytorch":
            ckpt = torch.load(checkpoint, map_location="cpu")
            state_dict = OrderedDict([(k.replace("module.", ""), v) for k, v in ckpt["model_state_dict"].items()])
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")
        elif checkpoint_type == "deepspeed":
            ckpt = torch.load(checkpoint, map_location="cpu")
            state_dict = ckpt["module"]
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")
        model_wrapper = ModelWrapper(model, text_tokenizer, image_processor)
        return model_wrapper

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None, required=True)
    parser.add_argument("--model_name", type=str, default="auto")
    parser.add_argument("--video_folder", type=str, default="auto")
    parser.add_argument("--checkpoint", type=str, default="auto")
    parser.add_argument("--checkpoint_type", type=str, default="deepspeed")
    parser.add_argument("--save_videos", action="store_true")
    parser.add_argument("--env_num", type=int, default=10)
    parser.add_argument("--total_env_num", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=600)
    parser.add_argument("--task_order", type=int, default=0)
    args = parser.parse_args()
    if args.model_name == "auto":
        model_name = args.log_dir.split("/")[-1].split("-")
        if len(model_name) == 1:
            args.model_name = model_name[0]
        else:
            args.model_name = "-".join((args.log_dir.split("/")[-1].split("-")[:-1]))
    if args.video_folder == "auto":
        assert args.checkpoint == "auto"
        args.video_folder = os.path.join(args.log_dir, "video", "latest")
    if args.checkpoint == "auto":
        with open(os.path.join(args.log_dir, "ckpt", "latest"), "r") as f:
            args.checkpoint = f.read().strip()
        args.checkpoint_type = "deepspeed"
        ckpt_name = os.listdir(os.path.join(args.log_dir, "ckpt", args.checkpoint))[0]
        args.checkpoint = os.path.join(args.log_dir, "ckpt", args.checkpoint, ckpt_name)
    return args

def main():
    args = parse_args()
    seed_all(args.seed)

    with open(f"{args.log_dir}/eval_cmd.txt", "w") as f:
        f.write("python " + " ".join(sys.argv) + "\n")
        f.write(json.dumps(vars(args), indent=4) + "\n")

    model = create_model(model_name=args.model_name, checkpoint_type=args.checkpoint_type, checkpoint=args.checkpoint)

    success_rates = {}
    for benchmark_name in BENCHMARKS:
        benchmark = get_benchmark(benchmark_name)(args.task_order)
        for task_id in range(benchmark.n_tasks):
            video_folder = os.path.join(args.video_folder, benchmark_name, f"task_{task_id}")
            os.makedirs(video_folder, exist_ok=True)
            print(f"=> [info] eval {benchmark_name} {task_id}/{benchmark.n_tasks}")
            success_rate = eval_env(
                model, 
                benchmark, 
                task_id, 
                video_folder, 
                args.save_videos, 
                args.env_num, 
                args.seed, 
                args.max_steps,
                args.total_env_num
            )
            success_rates[f"{benchmark_name}_{task_id}"] = success_rate
            with open(os.path.join(args.log_dir, "eval_results.txt"), "w") as f:
                json.dump(success_rates, f, indent=4)
    print(success_rates)
    with open(os.path.join(args.log_dir, "eval_results.txt"), "w") as f:
        json.dump(success_rates, f, indent=4)

def eval_env(model, benchmark, task_id, video_folder, save_videos, env_num, seed, max_steps, total_env_num):
    task = benchmark.get_task(task_id)
    task_name = task.name
    # language = task.language
    language = " ".join(task_name.split("_"))

    init_states_path = os.path.join(
        get_libero_path("init_states"), 
        task.problem_folder, 
        task.init_states_file
    )
    init_states = torch.load(init_states_path)

    def _eval_env(indices):
        with VideoWriter2(indices[0], video_folder, save_videos, single_video=False) as video_writer:
            env_args = {
                "bddl_file_name": os.path.join(
                    get_libero_path("bddl_files"), 
                    task.problem_folder, 
                    task.bddl_file
                ),
                "camera_heights": 128,
                "camera_widths": 128,
                "camera_depths": True,
            }
            env = SubprocVectorEnv(
                [lambda: OffScreenRenderEnv(**env_args) for _ in range(len(indices))]
            )
            env.reset()
            env.seed(seed)
            model.reset()

            dones = [False] * len(indices)
            steps = 0
            init_states_ = init_states[indices]
            obs = env.set_init_state(init_states_)

            for _ in range(5):  # simulate the physics without any actions
                env.step(np.zeros((env_num, 7)))

            def _obs_fmt(raw_obs):
                new_obs = []
                for obs in raw_obs:
                    new_obs.append({
                        "rgb_obs": {
                            "rgb_static": obs["agentview_image"],
                            "rgb_gripper": obs["robot0_eye_in_hand_image"],
                        },
                        "depth_obs": {
                            "depth_static": depthimg2Meters(obs["agentview_depth"]),
                            "depth_gripper": depthimg2Meters(obs["robot0_eye_in_hand_depth"]),
                        },
                    })
                return new_obs

            with torch.no_grad(), tqdm(total=max_steps, ncols=100) as tbar:
                while steps < max_steps:
                    steps += 1
                    tbar.update(1)
                    actions = model.step_batch(_obs_fmt(obs), [language] * len(indices))
                    obs, reward, done, info = env.step(actions)
                    video_writer.append_vector_obs(
                        obs, dones, camera_name="agentview_image"
                    )

                    # check whether succeed
                    for k in range(len(indices)):
                        dones[k] = dones[k] or done[k]
                    
                    if all(dones):
                        break
                    num_success = 0
                    for k in range(len(indices)):
                        num_success += int(dones[k])
                    tbar.set_postfix(success_rate=f"{num_success} / {len(indices)}")
                num_success = 0
                for k in range(len(indices)):
                    num_success += int(dones[k])
            success_rate = num_success / len(indices)
            env.close()
            print(f"task: {task_name}, success_rate: {success_rate}")
            return success_rate

    total_success_num = 0
    env_num_count = 0
    for start_idx in range(0, init_states.shape[0], env_num):
        if start_idx + env_num <= init_states.shape[0]:
            indices = np.arange(env_num) + start_idx
        else:
            indices = np.arange(init_states.shape[0] - start_idx) + start_idx
        if total_env_num > 0 and indices.shape[0] + env_num_count > total_env_num:
            indices = indices[:total_env_num - env_num_count]
        env_num_count += indices.shape[0]
        if len(indices) == 0: break
        success_rate = _eval_env(indices)
        total_success_num = total_success_num + success_rate * len(indices)
    success_rate = total_success_num / env_num_count
    print(f"task: {task_name}, success_rate: {success_rate}")
    return success_rate

if __name__ == "__main__":
    main()
