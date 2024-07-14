import os
import sys
import json
import shutil
import logging
import argparse
import warnings
import functools
warnings.filterwarnings("ignore")
from typing import Optional

import torch
import open_clip
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.roboflamingo.helpers import PerceiverResampler
from datasets.depth_libero_dataset import load_DepthLiberoDataset
from main import DATASET_CONFIG, get_config

class FeaturePool():
    """
    This class implements a feature buffer that stores previously encoded features

    This buffer enables us to initialize the codebook using a history of generated features
    rather than the ones produced by the latest encoders
    """
    def __init__(self, pool_size, dim=64):
        """
        Initialize the FeaturePool class

        Parameters:
            pool_size(int) -- the size of featue buffer
        """
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.nums_features = 0
            self.features = (torch.rand((pool_size, dim)) * 2 - 1)/ pool_size

    def query(self, features):
        """
        return features from the pool
        """
        self.features = self.features.to(features.device)    
        if self.nums_features < self.pool_size:
            if features.size(0) > self.pool_size: # if the batch size is large enough, directly update the whole codebook
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
                self.nums_features = self.pool_size
            else:
                # if the mini-batch is not large nuough, just store it for the next update
                num = self.nums_features + features.size(0)
                self.features[self.nums_features:num] = features
                self.nums_features = num
        else:
            if features.size(0) > int(self.pool_size):
                random_feat_id = torch.randint(0, features.size(0), (int(self.pool_size),))
                self.features = features[random_feat_id]
            else:
                random_id = torch.randperm(self.pool_size)
                self.features[random_id[:features.size(0)]] = features

        return self.features

class VectorQuantizerEMA(nn.Module):  # reference CVQ-VAE: https://github.com/lyndonzheng/CVQ-VAE/blob/main/quantise.py

    """
    Improved version over vector quantiser, with the dynamic initialisation
    for these unoptimised "dead" points.
    num_embed: number of codebook entry
    embed_dim: dimensionality of codebook entry
    beta: weight for the commitment loss
    distance: distance for looking up the closest code
    anchor: anchor sampled methods
    first_batch: if true, the offline version of our model
    contras_loss: if true, use the contras_loss to further improve the performance
    """
    def __init__(self, num_embed, embed_dim, beta, distance='cos', 
                 anchor='probrandom', first_batch=False, contras_loss=False):
        super().__init__()

        self.num_embed = num_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.distance = distance
        self.anchor = anchor
        self.first_batch = first_batch
        self.contras_loss = contras_loss
        self.decay = 0.99
        self.init = False

        self.pool = FeaturePool(self.num_embed, self.embed_dim)
        self.embedding = nn.Embedding(self.num_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embed, 1.0 / self.num_embed)
        self.register_buffer("embed_prob", torch.zeros(self.num_embed))
        self.print_flag = True
    
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False, training=True):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.embed_dim)

        # clculate the distance
        if self.distance == 'l2':
            # l2 distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
            d = - torch.sum(z_flattened.detach() ** 2, dim=1, keepdim=True) - \
                torch.sum(self.embedding.weight ** 2, dim=1) + \
                2 * torch.einsum('bd, dn-> bn', z_flattened.detach(), rearrange(self.embedding.weight, 'n d-> d n'))
        elif self.distance == 'cos':
            # cosine distances from z to embeddings e_j 
            normed_z_flattened = F.normalize(z_flattened, dim=1).detach()
            normed_codebook = F.normalize(self.embedding.weight, dim=1)
            d = torch.einsum('bd,dn->bn', normed_z_flattened, rearrange(normed_codebook, 'n d -> d n'))

        # encoding
        sort_distance, indices = d.sort(dim=1)
        # look up the closest point for the indices
        encoding_indices = indices[:,-1]
        encodings = torch.zeros(encoding_indices.unsqueeze(1).shape[0], self.num_embed, device=z.device)
        encodings.scatter_(1, encoding_indices.unsqueeze(1), 1)

        # quantise and unflatten
        z_q = torch.matmul(encodings, self.embedding.weight).view(z.shape)
        # compute loss for embedding
        if training:
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # count
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        min_encodings = encodings

        # online clustered reinitialisation for unoptimized points
        if training:
            # calculate the average usage of code entries
            self.embed_prob.mul_(self.decay).add_(avg_probs, alpha= 1 - self.decay)
            # running average updates
            if self.anchor in ['closest', 'random', 'probrandom'] and (not self.init):
                # closest sampling
                if self.anchor == 'closest':
                    sort_distance, indices = d.sort(dim=0)
                    random_feat = z_flattened.detach()[indices[-1,:]]
                # feature pool based random sampling
                elif self.anchor == 'random':
                    random_feat = self.pool.query(z_flattened.detach())
                # probabilitical based random sampling
                elif self.anchor == 'probrandom':
                    norm_distance = F.softmax(d.t(), dim=1)
                    prob = torch.multinomial(norm_distance, num_samples=1).view(-1)
                    random_feat = z_flattened.detach()[prob]
                # decay parameter based on the average usage
                decay = torch.exp(-(self.embed_prob*self.num_embed*10)/(1-self.decay)-1e-3).unsqueeze(1).repeat(1, self.embed_dim)
                self.embedding.weight.data = self.embedding.weight.data * (1 - decay) + random_feat * decay
                if self.first_batch:
                    self.init = True
            # contrastive loss
            if self.contras_loss:
                sort_distance, indices = d.sort(dim=0)
                dis_pos = sort_distance[-max(1, int(sort_distance.size(0)/self.num_embed)):,:].mean(dim=0, keepdim=True)
                dis_neg = sort_distance[:int(sort_distance.size(0)*1/2),:]
                dis = torch.cat([dis_pos, dis_neg], dim=0).t() / 0.07
                contra_loss = F.cross_entropy(dis, torch.zeros((dis.size(0),), dtype=torch.long, device=dis.device))
                loss +=  contra_loss
        return z_q, encoding_indices, loss, perplexity

class DepthModel(nn.Module):

    def __init__(self, depth_sensor_encoder: nn.Module, embedding_dim: int) -> None:
        super().__init__()
        self.depth_sensor_encoder = depth_sensor_encoder
        self.embedding_dim = embedding_dim
        self.depth_perceiver = PerceiverResampler(dim=self.embedding_dim)

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

    def forward(self, depth_static: torch.Tensor, depth_gripper: torch.Tensor):
        """
        depth_static: (B, T, 1, 1, 3, 224, 224)
        depth_gripper: (B, T, 1, 1, 3, 224, 224)
        """
        B = depth_static.shape[0]
        depth_static = depth_static.reshape(-1, *depth_static.shape[2:])  # (B*T, 1, 1, 3, 224, 224)
        depth_gripper = depth_gripper.reshape(-1, *depth_gripper.shape[2:])  # (B*T, 1, 1, 3, 224, 224)
        depth_static = self._encode_depth_sensor(depth_static)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
        depth_gripper = self._encode_depth_sensor(depth_gripper)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
        depth_static = self.depth_perceiver(depth_static)  # (B * window_size, 1, num_latents, depth_embedding_dim), num_latents = 64
        depth_gripper = self.depth_perceiver(depth_gripper)  # (B * window_size, 1, num_latents, depth_embedding_dim), num_latents = 64
        return depth_static, depth_gripper

def parse_args():
    parser = argparse.ArgumentParser(description="Train VQ model")
    parser.add_argument(
        "--drf_model_path",
        type=str,
        default="runs/roboflamingo-mpt_3b_depth-DepthLiberoDataset/ckpt/global_step22645/mp_rank_00_model_states.pt",
        help="Path to the pretrained DRF model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for dataloader"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Number of epochs"
    )
    parser.add_argument("--dataset_name", type=str, default="DepthLiberoDataset")
    parser.add_argument("--log_dir", type=str, default="runs/vq", help="Log directory")

    return parser.parse_args()

def get_logger(
    filename: str,
    logger_name: Optional[str] = None,
    log_level: int = logging.INFO,
    fmt_str: str = "[%(asctime)s] [%(levelname)s] %(message)s",
) -> logging.Logger:
    logging.basicConfig(filename=filename, filemode='w', format=fmt_str, level=log_level)
    return logging

def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(drf_model_path, device="cuda"):
    vision_encoder_path = "ViT-L-14"
    vision_encoder_pretrained = "openai"
    vision_encoder_cache_dir = "RoboFlamingo/models/clip"
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        model_name=vision_encoder_path,
        pretrained=vision_encoder_pretrained,
        cache_dir=vision_encoder_cache_dir,
    )
    vision_encoder.visual.output_tokens = True
    depth_model = DepthModel(vision_encoder, open_clip.get_model_config(vision_encoder_path)["vision_cfg"]["width"])
    missing_keys, unexpected_keys = depth_model.load_state_dict(torch.load(drf_model_path, map_location="cpu")["module"], strict=False)
    print(f"unexpected_keys: {unexpected_keys}")
    print(f"missing_keys: {missing_keys}")
    depth_model = depth_model.to(device)
    depth_model.eval()
    depth_model.requires_grad_(False)

    codebook = VectorQuantizerEMA(num_embed=512, embed_dim=open_clip.get_model_config(vision_encoder_path)["vision_cfg"]["width"], beta=0.25).to(device)

    return depth_model, codebook, image_processor

def load_dataset(key, image_processor):
    dataset_config = get_config(key, DATASET_CONFIG)
    if dataset_config["load_fn"] == "load_DepthLiberoDataset":
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
    else:
        raise NotImplementedError

def main():
    seed_everything(42)
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.join(args.log_dir, "ckpt"), exist_ok=True)
    if os.path.exists(os.path.join(args.log_dir, "tb")):
        shutil.rmtree(os.path.join(args.log_dir, "tb"))
        os.makedirs(os.path.join(args.log_dir, "tb"))
    with open(os.path.join(args.log_dir, "cmd.txt"), "w") as fo:
        fo.write("python " + " ".join(sys.argv) + "\n")
        fo.write(json.dumps(vars(args), indent=4) + "\n")
    # create logger
    logger = get_logger(os.path.join(args.log_dir, "train.log"), logger_name="train")
    # create depth model and load pretrained DRF model
    depth_model, codebook, image_processor = load_model(args.drf_model_path, args.device)

    # create dataset&dataloader
    dataset = load_dataset(args.dataset_name, image_processor)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        collate_fn=dataset.collater
    )

    optimizer = optim.Adam(codebook.parameters(), lr=args.learning_rate, amsgrad=True)
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, "tb"))
    for epoch_i in range(args.num_epochs):
        iter_i = 0
        for batch in (pbar := tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch_i}/{args.num_epochs}")):
            depth_static = batch["depth_static_tensors"]
            depth_gripper = batch["depth_gripper_tensors"]
            depth_static = (depth_static.to(args.device).unsqueeze(2).unsqueeze(2))  # (B, window_size, 1, 1, 3, 224, 224)
            depth_gripper = (depth_gripper.to(args.device).unsqueeze(2).unsqueeze(2))  # (B, window_size, 1, 1, 3, 224, 224)
            depth_static = depth_static[:, np.random.randint(0, 6)::6]
            depth_gripper = depth_gripper[:, np.random.randint(0, 6)::6]
            with torch.no_grad():
                depth_static, depth_gripper = depth_model(depth_static, depth_gripper)
            quantized_static, indices_static, commit_loss_static, perplexity_static = codebook(depth_static)
            quantized_gripper, indices_gripper, commit_loss_gripper, perplexity_gripper = codebook(depth_gripper)
            reconstruction_loss = torch.mean((quantized_static - depth_static) ** 2) + torch.mean((quantized_gripper - depth_gripper) ** 2)
            commit_loss = commit_loss_static + commit_loss_gripper
            loss = reconstruction_loss + commit_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            log_msg = f"Epoch {epoch_i}, Iter {iter_i}, Loss: total={loss.item()} reconstruction={reconstruction_loss.item()} commit={commit_loss.item()}, Perplexity: static={perplexity_static} gripper={perplexity_gripper}"
            pbar.set_description_str(f"loss: {loss.item():.4f}, recon: {reconstruction_loss.item():.4f}, commit: {commit_loss.item():.4f}, p_static: {perplexity_static:.4f}, p_gripper: {perplexity_gripper:.4f}")
            logger.info(log_msg)
            writer.add_scalar("Loss/total", loss.item(), epoch_i * len(dataloader) + iter_i)
            writer.add_scalar("Loss/reconstruction", reconstruction_loss.item(), epoch_i * len(dataloader) + iter_i)
            writer.add_scalar("Loss/commit", commit_loss.item(), epoch_i * len(dataloader) + iter_i)
            writer.add_scalar("Perplexity/static", perplexity_static, epoch_i * len(dataloader) + iter_i)
            writer.add_scalar("Perplexity/gripper", perplexity_gripper, epoch_i * len(dataloader) + iter_i)
            iter_i += 1
        torch.save(codebook.state_dict(), os.path.join(args.log_dir, "ckpt", f"epoch{epoch_i}.pt"))

if __name__ == "__main__":
    main()
