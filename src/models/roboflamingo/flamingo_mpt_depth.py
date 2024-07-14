"""
Origin Repo: https://github.com/RoboFlamingo/RoboFlamingo
"""
import os
import copy
from pathlib import Path
from functools import partial
from typing import Optional, List
from collections import namedtuple

import torch
import deepspeed.comm.comm as dcomm
import open_clip
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.bc_transformer.policy_head import GMMHead
from models.roboflamingo.helpers import PerceiverResampler, PerceiverAttention, FeedForward, NoLatentPerceiverResampler
from models.roboflamingo.flamingo_mpt import MPTFlamingo
from models.roboflamingo.flamingo_lm import FlamingoLMMixin
from models.roboflamingo.flamingo_utils import extend_instance

class LinearResampler(nn.Module):

    def __init__(self, input_latents: int, num_latents: int) -> None:
        super().__init__()
        self.input_latents = input_latents
        self.num_latents = num_latents
        self.proj = nn.Linear(input_latents, num_latents)

    def forward(self, x):
        """
        x.shape = (B * window_size, 1, 1, input_latents, vision_embedding_dim)
        return.shape = (B * window_size, 1, num_latents, vision_embedding_dim)
        """
        x = x.transpose(-1, -2)  # (B * window_size, 1, 1, vision_embedding_dim, input_latents)
        x = self.proj(x)  # (B * window_size, 1, 1, vision_embedding_dim, num_latents)
        x = x.transpose(-1, -2)  # (B * window_size, 1, 1, num_latents, vision_embedding_dim)
        return x[:, 0]  # (B * window_size, 1, num_latents, vision_embedding_dim)

class VectorQuantizer(nn.Module):
    """
    Discretization bottleneck part of the VQ-VAE.

    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z: torch.Tensor):
        """
        Inputs the output of the encoder network z and maps it to a discrete 
        one-hot vector that is the index of the closest embedding vector e_j

        z (continuous) -> z_q (discrete)

        z.shape = (..., E)

        """
        z_flattened = z.view(-1, self.e_dim)  # (B, E)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())

        # find closest encodings
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(
            min_encoding_indices.shape[0], self.n_e).to(z.device)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
            torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        return loss, z_q, perplexity, min_encodings, min_encoding_indices

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

    @torch.no_grad()
    def concat_all_gather(self, tensor):

        tensors_gather = [
            torch.ones_like(tensor) for _ in range(dcomm.get_world_size())
        ]

        # dcomm.all_gather_into_tensor(tensors_gather, tensor, async_op=False)
        # output = torch.cat(tensors_gather, dim=0)

        tensors_gather = torch.cat(tensors_gather, dim=0)
        dcomm.all_gather_into_tensor(tensors_gather, tensor, async_op=False)
        output = tensors_gather
        return output
    
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
        else:
            if self.print_flag:
                self.print_flag = False
                print("落魄了, 家人们, 以后不再更新Codebook了")
            loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q.detach() - z.detach()) ** 2)
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()
        # count
        # count
        if training:
            encodings_gather = self.concat_all_gather(encodings)
            avg_probs = torch.mean(encodings_gather, dim=0)
        else:
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
                dcomm.broadcast(self.embedding.weight.data, src=0)
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
        return loss, z_q, perplexity, min_encodings, encoding_indices

class DepthMPTFlamingo(MPTFlamingo):

    def __init__(
        self, 
        vision_encoder: nn.Module,
        depth_sensor_encoder: nn.Module,
        lang_encoder: nn.Module, 
        media_token_id: int, 
        vision_embedding_dim: int, 
        depth_embedding_dim: int,
        cross_attn_every_n_layers: int, 
        window_size: int,
        resampler_type: str = "perceiver_resampler",
    ) -> None:
        super().__init__(
            vision_encoder, 
            lang_encoder, 
            media_token_id, 
            vision_embedding_dim, 
            cross_attn_every_n_layers, 
            window_size
        )
        self.depth_sensor_encoder = depth_sensor_encoder
        self.depth_embedding_dim = depth_embedding_dim
        self.resampler_type = resampler_type
        if resampler_type == "perceiver_resampler":  # two_perceiver_resampler_late_fusion
            self.depth_perceiver = PerceiverResampler(dim=self.depth_embedding_dim)
        elif resampler_type == "pred_depth":  # 这里的resampler_type不会去修改模型的resampler_type，只是用于区分不同的模型
            print("PRED DEPTH!!!!!!!!")
            self.depth_perceiver = PerceiverResampler(dim=self.depth_embedding_dim)
            self.depth_pred = PerceiverResampler(dim=self.depth_embedding_dim)
        elif resampler_type == "depth_codebook":
            print("WITH VQ CODEBOOK!!!!!!!!")
            self.depth_perceiver = PerceiverResampler(dim=self.depth_embedding_dim)
            n_e = int(os.environ.get("NUM_EMBEDDING", "512"))
            print("NUM_EMBEDDING", n_e)
            self.depth_vq = VectorQuantizer(n_e=n_e, e_dim=self.depth_embedding_dim, beta=0.25)
        elif resampler_type == "depth_codebook_pred_depth":
            print("WITH VQ CODEBOOK&PRED DEPTH!!!!!!!!")
            self.depth_perceiver = PerceiverResampler(dim=self.depth_embedding_dim)
            self.depth_vq = VectorQuantizer(n_e=512, e_dim=self.depth_embedding_dim, beta=0.25)
            self.depth_pred = PerceiverResampler(dim=self.depth_embedding_dim)
        elif resampler_type == "depth_codebook_ema":
            print("WITH VQ CODEBOOK EMA!!!!!!!!")
            self.depth_perceiver = PerceiverResampler(dim=self.depth_embedding_dim)
            self.depth_vq = VectorQuantizerEMA(num_embed=512, embed_dim=self.depth_embedding_dim, beta=0.25)
        elif resampler_type == "depth_codebook_ema_finetune":
            print("WITH VQ CODEBOOK EMA FINETUNE!!!!!!!!")
            self.depth_perceiver = PerceiverResampler(dim=self.depth_embedding_dim)
            self.depth_vq = VectorQuantizerEMA(num_embed=512, embed_dim=self.depth_embedding_dim, beta=0.25)
        elif resampler_type == "depth_codebook_ema_pred_depth":
            print("WITH VQ CODEBOOK EMA&PRED DEPTH!!!!!!!!")
            self.depth_perceiver = PerceiverResampler(dim=self.depth_embedding_dim)
            self.depth_vq = VectorQuantizerEMA(num_embed=512, embed_dim=self.depth_embedding_dim, beta=0.25)
            self.depth_pred = PerceiverResampler(dim=self.depth_embedding_dim)
        elif resampler_type == "KD":
            print("WITH KD!!!!!!!!")
            self.depth_perceiver = PerceiverResampler(dim=self.depth_embedding_dim)
        elif resampler_type == "MM_Prompt":
            print("WITH MM PROMPT!!!!!!!!")
            self.depth_perceiver = PerceiverResampler(dim=self.depth_embedding_dim)
            self.prompt = nn.Parameter(torch.randn(3, self.depth_embedding_dim))
        else:
            raise NotImplementedError

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
    
    def _encode_multi_sensor_post_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, depth_static: torch.Tensor, depth_gripper: torch.Tensor):
        if vision_rgb is not None and vision_gripper is not None:
            vision_rgb = self._encode_vision(vision_rgb)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
            vision_gripper = self._encode_vision(vision_gripper)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
            vision_rgb = self.perceiver(vision_rgb)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
            vision_gripper = self.perceiver(vision_gripper)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
        if depth_static is not None and depth_gripper is not None:
            depth_static = self._encode_depth_sensor(depth_static)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
            depth_gripper = self._encode_depth_sensor(depth_gripper)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
            depth_static = self.depth_perceiver(depth_static)  # (B * window_size, 1, num_latents, depth_embedding_dim), num_latents = 64
            depth_gripper = self.depth_perceiver(depth_gripper)  # (B * window_size, 1, num_latents, depth_embedding_dim), num_latents = 64
        vision_x = []
        if vision_rgb is not None and vision_gripper is not None:
            vision_x.append(vision_rgb)
            vision_x.append(vision_gripper)
        if depth_static is not None and depth_gripper is not None:
            vision_x.append(depth_static)
            vision_x.append(depth_gripper)
        vision_x = torch.cat(vision_x, dim=2)  # (B * window_size, 1, 4 * num_latents, vision_embedding_dim + vision_embedding_dim)
        for layer in self.lang_encoder._get_decoder_layers():
            # the function `condition_vis_x` is defined in `models.flamingo_lm.FlamingoLayer`
            layer.condition_vis_x(vision_x)
        return vision_rgb, vision_gripper, depth_static, depth_gripper

    def _encode_multi_sensor_one_perceiver_resampler_late_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, depth_static: torch.Tensor, depth_gripper: torch.Tensor):
        vision_rgb = self._encode_vision(vision_rgb)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
        vision_gripper = self._encode_vision(vision_gripper)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
        vision_rgb = self.perceiver(vision_rgb)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
        vision_gripper = self.perceiver(vision_gripper)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
        depth_static = self._encode_depth_sensor(depth_static)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
        depth_gripper = self._encode_depth_sensor(depth_gripper)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
        depth_static = self.perceiver(depth_static)  # (B * window_size, 1, num_latents, depth_embedding_dim), num_latents = 64
        depth_gripper = self.perceiver(depth_gripper)  # (B * window_size, 1, num_latents, depth_embedding_dim), num_latents = 64
        vision_x = torch.cat([vision_rgb, vision_gripper, depth_static, depth_gripper], dim=2)  # (B * window_size, 1, 4 * num_latents, vision_embedding_dim + vision_embedding_dim)
        for layer in self.lang_encoder._get_decoder_layers():
            # the function `condition_vis_x` is defined in `models.flamingo_lm.FlamingoLayer`
            layer.condition_vis_x(vision_x)
        return vision_rgb, vision_gripper, depth_static, depth_gripper

    def _encode_multi_sensor_one_perceiver_resampler_early_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, depth_static: torch.Tensor, depth_gripper: torch.Tensor):
        vision_rgb = self._encode_vision(vision_rgb)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
        vision_gripper = self._encode_vision(vision_gripper)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
        depth_static = self._encode_depth_sensor(depth_static)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
        depth_gripper = self._encode_depth_sensor(depth_gripper)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
        vision_x = self.perceiver(
            torch.cat(
                [vision_rgb, vision_gripper, depth_static, depth_gripper],
                dim=3
            )
        )  # (B * window_size, 1, num_latents, vision_embedding_dim)
        for layer in self.lang_encoder._get_decoder_layers():
            # the function `condition_vis_x` is defined in `models.flamingo_lm.FlamingoLayer`
            layer.condition_vis_x(vision_x)
        return vision_rgb, vision_gripper, depth_static, depth_gripper

    def _encode_multi_sensor_bottleneck(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, depth_static: torch.Tensor, depth_gripper: torch.Tensor):
        vision_rgb = self._encode_vision(vision_rgb)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
        vision_gripper = self._encode_vision(vision_gripper)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
        depth_static = self._encode_depth_sensor(depth_static)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
        depth_gripper = self._encode_depth_sensor(depth_gripper)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
        modality0_model = self.perceiver
        modality1_model = self.depth_perceiver
        modality0 = [vision_rgb, vision_gripper]
        modality1 = [depth_static, depth_gripper]
        b, T = vision_rgb.shape[:2]

        # prepare each modality
        def _vmap(x, frame_embs, media_time_embs):
            b, T, F, v = x.shape[:4]
            # frame and media time embeddings
            if frame_embs is not None:
                frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
                x = x + frame_embs
            x = rearrange(
                x, "b T F v d -> b T (F v) d"
            )  # flatten the frame and spatial dimensions
            if media_time_embs is not None:
                x = x + self.media_time_embs[:T]
            return x
        modality0 = list(map(
            partial(_vmap, frame_embs=modality0_model.frame_embs, media_time_embs=modality0_model.media_time_embs),
            modality0
        ))  # [(batch_size * window_size, 1, patch_num, vision_embedding_dim)]
        modality1 = list(map(
            partial(_vmap, frame_embs=modality1_model.frame_embs, media_time_embs=modality1_model.media_time_embs),
            modality1
        ))  # [(batch_size * window_size, 1, patch_num, vision_embedding_dim)]

        # bottleneck_fusion
        modality0_latents = repeat(modality0_model.latents, "n d -> b T n d", b=b, T=T)
        modality0_latents = [modality0_latents] * len(modality0)
        modality1_latents = repeat(modality1_model.latents, "n d -> b T n d", b=b, T=T)
        modality1_latents = [modality1_latents] * len(modality1)
        bottleneck_latents = repeat(self.bottleneck_latents, "n d -> b T n d", b=b, T=T)

        def _fusion(attn, ff, q, kv):
            q = attn(kv, q) + q
            q = ff(q) + q
            return q

        for i in range(len(modality0_model.layers)):
            modality0_attn, modality0_ff = modality0_model.layers[i]
            modality1_attn, modality1_ff = modality1_model.layers[i]
            bottleneck_attn, bottleneck_ff = self.bottleneck_layers[i]
            modality0_latents_new = [
                _fusion(
                    attn=modality0_attn,
                    ff=modality0_ff,
                    q=modality0_latents[i],
                    kv=torch.cat([
                        modality0[i],
                        bottleneck_latents
                    ], dim=-2)
                )
                for i in range(len(modality0_latents))
            ]
            modality1_latents_new = [
                _fusion(
                    attn=modality1_attn,
                    ff=modality1_ff,
                    q=modality1_latents[i],
                    kv=torch.cat([
                        modality1[i],
                        bottleneck_latents
                    ], dim=-2)
                )
                for i in range(len(modality1_latents))
            ]
            bottleneck_latents_new = _fusion(
                attn=bottleneck_attn,
                ff=bottleneck_ff,
                q=bottleneck_latents,
                kv=torch.cat(modality0 + modality1, dim=-2)
            )
            modality0_latents = modality0_latents_new
            modality1_latents = modality1_latents_new
            bottleneck_latents = bottleneck_latents_new
        
        modality0_latents = list(map(
            lambda x: modality0_model.norm(x),
            modality0_latents
        ))
        modality1_latents = list(map(
            lambda x: modality1_model.norm(x),
            modality1_latents
        ))
        bottleneck_latents = self.bottleneck_norm(bottleneck_latents)
        vision_x = torch.cat(modality0_latents + modality1_latents, dim=2)  # (B * window_size, 1, 4 * num_latents, vision_embedding_dim + vision_embedding_dim)
        for layer in self.lang_encoder._get_decoder_layers():
            # the function `condition_vis_x` is defined in `models.flamingo_lm.FlamingoLayer`
            layer.condition_vis_x(vision_x)
        return modality0_latents[0], modality1_latents[0], modality0_latents[1], modality1_latents[1], bottleneck_latents

    def _encode_multi_sensor_bottleneck_with_probalitiy_mask(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, depth_static: torch.Tensor, depth_gripper: torch.Tensor):
        vision_rgb = self._encode_vision(vision_rgb)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
        vision_gripper = self._encode_vision(vision_gripper)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
        depth_static = self._encode_depth_sensor(depth_static)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
        depth_gripper = self._encode_depth_sensor(depth_gripper)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
        modality0_model = self.perceiver
        modality1_model = self.depth_perceiver
        modality0 = [vision_rgb, vision_gripper]
        modality1 = [depth_static, depth_gripper]
        b, T = vision_rgb.shape[:2]

        # prepare each modality
        def _vmap(x, frame_embs, media_time_embs):
            b, T, F, v = x.shape[:4]
            # frame and media time embeddings
            if frame_embs is not None:
                frame_embs = repeat(self.frame_embs[:F], "F d -> b T F v d", b=b, T=T, v=v)
                x = x + frame_embs
            x = rearrange(
                x, "b T F v d -> b T (F v) d"
            )  # flatten the frame and spatial dimensions
            if media_time_embs is not None:
                x = x + self.media_time_embs[:T]
            return x
        modality0 = list(map(
            partial(_vmap, frame_embs=modality0_model.frame_embs, media_time_embs=modality0_model.media_time_embs),
            modality0
        ))  # [(batch_size * window_size, 1, patch_num, vision_embedding_dim)]
        modality1 = list(map(
            partial(_vmap, frame_embs=modality1_model.frame_embs, media_time_embs=modality1_model.media_time_embs),
            modality1
        ))  # [(batch_size * window_size, 1, patch_num, vision_embedding_dim)]

        # bottleneck_fusion
        modality0_latents = repeat(modality0_model.latents, "n d -> b T n d", b=b, T=T)
        modality0_latents = [modality0_latents] * len(modality0)
        modality1_latents = repeat(modality1_model.latents, "n d -> b T n d", b=b, T=T)
        modality1_latents = [modality1_latents] * len(modality1)
        bottleneck_latents = repeat(self.bottleneck_latents, "n d -> b T n d", b=b, T=T)

    def _encode_multi_sensor_with_vq(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, depth_static: torch.Tensor, depth_gripper: torch.Tensor):
        if vision_rgb is not None and vision_gripper is not None:
            vision_rgb = self._encode_vision(vision_rgb)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
            vision_gripper = self._encode_vision(vision_gripper)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
            vision_rgb = self.perceiver(vision_rgb)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
            vision_gripper = self.perceiver(vision_gripper)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
        if depth_static is not None and depth_gripper is not None:
            depth_static = self._encode_depth_sensor(depth_static)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
            depth_gripper = self._encode_depth_sensor(depth_gripper)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
            depth_static = self.depth_perceiver(depth_static)  # (B * window_size, 1, num_latents, depth_embedding_dim), num_latents = 64
            depth_gripper = self.depth_perceiver(depth_gripper)  # (B * window_size, 1, num_latents, depth_embedding_dim), num_latents = 64
            depth_static_before_codebook = depth_static
            depth_gripper_before_codebook = depth_gripper
            loss_static, depth_static, perplexity_static, _, indices_static = self.depth_vq(depth_static)
            loss_gripper, depth_gripper, perplexity_gripper, _, indices_gripper = self.depth_vq(depth_gripper)
        vision_x = []
        if vision_rgb is not None and vision_gripper is not None:
            vision_x.append(vision_rgb)
            vision_x.append(vision_gripper)
        if depth_static is not None and depth_gripper is not None:
            vision_x.append(depth_static)
            vision_x.append(depth_gripper)
        vision_x = torch.cat(vision_x, dim=2)  # (B * window_size, 1, 4 * num_latents, vision_embedding_dim + vision_embedding_dim)
        for layer in self.lang_encoder._get_decoder_layers():
            # the function `condition_vis_x` is defined in `models.flamingo_lm.FlamingoLayer`
            layer.condition_vis_x(vision_x)
        return vision_rgb, vision_gripper, depth_static, depth_gripper, loss_static, perplexity_static, loss_gripper, perplexity_gripper, indices_static, indices_gripper, \
            depth_static_before_codebook, depth_gripper_before_codebook

    def _encode_multi_sensor_with_vq_ema(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, depth_static: torch.Tensor, depth_gripper: torch.Tensor, mode: bool):
        if vision_rgb is not None and vision_gripper is not None:
            vision_rgb = self._encode_vision(vision_rgb)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
            vision_gripper = self._encode_vision(vision_gripper)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
            vision_rgb = self.perceiver(vision_rgb)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
            vision_gripper = self.perceiver(vision_gripper)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
        if depth_static is not None and depth_gripper is not None:
            depth_static = self._encode_depth_sensor(depth_static)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
            depth_gripper = self._encode_depth_sensor(depth_gripper)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
            depth_static = self.depth_perceiver(depth_static)  # (B * window_size, 1, num_latents, depth_embedding_dim), num_latents = 64
            depth_gripper = self.depth_perceiver(depth_gripper)  # (B * window_size, 1, num_latents, depth_embedding_dim), num_latents = 64
            depth_static_before_codebook = depth_static
            depth_gripper_before_codebook = depth_gripper
            loss_static, depth_static, perplexity_static, _, indices_static = self.depth_vq(depth_static, training=(mode=="training"))
            loss_gripper, depth_gripper, perplexity_gripper, _, indices_gripper = self.depth_vq(depth_gripper, training=(mode=="training"))
        vision_x = []
        if vision_rgb is not None and vision_gripper is not None:
            vision_x.append(vision_rgb)
            vision_x.append(vision_gripper)
        if depth_static is not None and depth_gripper is not None:
            vision_x.append(depth_static)
            vision_x.append(depth_gripper)
        vision_x = torch.cat(vision_x, dim=2)  # (B * window_size, 1, 4 * num_latents, vision_embedding_dim + vision_embedding_dim)
        for layer in self.lang_encoder._get_decoder_layers():
            # the function `condition_vis_x` is defined in `models.flamingo_lm.FlamingoLayer`
            layer.condition_vis_x(vision_x)
        return vision_rgb, vision_gripper, depth_static, depth_gripper, loss_static, perplexity_static, loss_gripper, perplexity_gripper, indices_static, indices_gripper, \
            depth_static_before_codebook, depth_gripper_before_codebook

    def _encode_multi_sensor_post_fusion_with_prompt(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor, depth_static: torch.Tensor, depth_gripper: torch.Tensor):
        prompt_type = 0
        if vision_rgb is not None and vision_gripper is not None:
            prompt_type += 1
            vision_rgb = self._encode_vision(vision_rgb)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
            vision_gripper = self._encode_vision(vision_gripper)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
            vision_rgb = self.perceiver(vision_rgb)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
            vision_gripper = self.perceiver(vision_gripper)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
        if depth_static is not None and depth_gripper is not None:
            prompt_type += 2
            depth_static = self._encode_depth_sensor(depth_static)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
            depth_gripper = self._encode_depth_sensor(depth_gripper)  # (B * window_size, 1, 1, patch_num, depth_embedding_dim)
            depth_static = self.depth_perceiver(depth_static)  # (B * window_size, 1, num_latents, depth_embedding_dim), num_latents = 64
            depth_gripper = self.depth_perceiver(depth_gripper)  # (B * window_size, 1, num_latents, depth_embedding_dim), num_latents = 64

        prompt_type -= 1
        assert prompt_type < 3 and prompt_type >= 0, "At least one modality should be present"

        vision_x = []
        if vision_rgb is not None and vision_gripper is not None:
            vision_x.append(vision_rgb)
            vision_x.append(vision_gripper)
        if depth_static is not None and depth_gripper is not None:
            vision_x.append(depth_static)
            vision_x.append(depth_gripper)
        vision_x = torch.cat(vision_x, dim=2)  # (B * window_size, 1, 4 * num_latents, vision_embedding_dim)
        tgt_shape = list(vision_x.shape)
        tgt_shape[2] = 1
        prompt = self.prompt[prompt_type:prompt_type+1].expand(*tgt_shape)
        vision_x = torch.cat([prompt, vision_x], dim=2)

        for layer in self.lang_encoder._get_decoder_layers():
            # the function `condition_vis_x` is defined in `models.flamingo_lm.FlamingoLayer`
            layer.condition_vis_x(vision_x)
        return vision_rgb, vision_gripper, depth_static, depth_gripper

    def forward(
        self, 
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_gripper: torch.Tensor,
        depth_static: torch.Tensor, 
        depth_gripper: torch.Tensor,
        *args,
        **kwargs
    ):
        if vision_x is not None:
            B = vision_x.shape[0]
            vision_x = vision_x.reshape(-1, *vision_x.shape[2:])
            vision_gripper = vision_gripper.reshape(-1, *vision_gripper.shape[2:])
        if depth_static is not None:
            B = depth_static.shape[0]
            depth_static = depth_static.reshape(-1, *depth_static.shape[2:])
            depth_gripper = depth_gripper.reshape(-1, *depth_gripper.shape[2:])
        lang_x = lang_x.reshape(-1, *lang_x.shape[2:])
        attention_mask = attention_mask.reshape(-1, *attention_mask.shape[2:])
        if self.resampler_type == "perceiver_resampler":
            vision_x, vision_gripper, depth_static, depth_gripper = self._encode_multi_sensor_post_fusion(vision_x, vision_gripper, depth_static, depth_gripper)
        elif self.resampler_type == "pred_depth":
            if kwargs.get("mode", "training") == "training":  # training
                vision_x, vision_gripper, depth_static, depth_gripper = self._encode_multi_sensor_post_fusion(vision_x, vision_gripper, depth_static, depth_gripper)
                # vision_x: (B * window_size, 1, num_latents, vision_embedding_dim)
                # vision_gripper: (B * window_size, 1, num_latents, vision_embedding_dim)
                # depth_static: (B * window_size, 1, num_latents, depth_embedding_dim)
                # depth_gripper: (B * window_size, 1, num_latents, depth_embedding_dim)
                depth_static_pred = self.depth_pred(vision_x.unsqueeze(dim=1))
                depth_gripper_pred = self.depth_pred(vision_gripper.unsqueeze(dim=1))

                depth_pred_loss = (torch.nn.functional.mse_loss(depth_static_pred, depth_static, reduction="mean")
                                + torch.nn.functional.mse_loss(depth_gripper_pred, depth_gripper, reduction="mean")) / 2
            elif kwargs.get("mode", "training") == "vis":
                vision_x = self._encode_vision(vision_x)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
                vision_gripper = self._encode_vision(vision_gripper)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
                vision_x = self.perceiver(vision_x)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
                vision_gripper = self.perceiver(vision_gripper)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
                depth_static = self._encode_depth_sensor(depth_static)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
                depth_gripper = self._encode_depth_sensor(depth_gripper)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
                depth_static = self.depth_perceiver(depth_static)
                depth_gripper = self.depth_perceiver(depth_gripper)
                depth_static_pred = self.depth_pred(vision_x.unsqueeze(dim=1), vis_attn=True)
                last_attn = self.depth_pred.last_attn
                depth_gripper_pred = self.depth_pred(vision_gripper.unsqueeze(dim=1))
                diff = {
                    "static": torch.nn.functional.mse_loss(depth_static_pred, depth_static, reduction="mean"),
                    "gripper": torch.nn.functional.mse_loss(depth_gripper_pred, depth_gripper, reduction="mean")
                }
                vision = []
                vision.append(vision_x)
                vision.append(vision_gripper)
                vision.append(depth_static_pred)
                vision.append(depth_gripper_pred)
                vision = torch.cat(vision, dim=2)  # (B * window_size, 1, 4 * num_latents, vision_embedding_dim + vision_embedding_dim)
                depth_pred_loss = 0.0
                for layer in self.lang_encoder._get_decoder_layers():
                    # the function `condition_vis_x` is defined in `models.flamingo_lm.FlamingoLayer`
                    layer.condition_vis_x(vision)
            else:  # inference
                vision_x = self._encode_vision(vision_x)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
                vision_gripper = self._encode_vision(vision_gripper)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
                vision_x = self.perceiver(vision_x)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
                vision_gripper = self.perceiver(vision_gripper)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
                depth_static = self.depth_pred(vision_x.unsqueeze(dim=1))
                depth_gripper = self.depth_pred(vision_gripper.unsqueeze(dim=1))
                depth_static_pred = depth_static
                depth_gripper_pred = depth_gripper  # 为了和上面training的代码保持统一(尽管没什么用:-) )
                vision = []
                vision.append(vision_x)
                vision.append(vision_gripper)
                vision.append(depth_static)
                vision.append(depth_gripper)
                vision = torch.cat(vision, dim=2)  # (B * window_size, 1, 4 * num_latents, vision_embedding_dim + vision_embedding_dim)
                depth_pred_loss = 0.0
                for layer in self.lang_encoder._get_decoder_layers():
                    # the function `condition_vis_x` is defined in `models.flamingo_lm.FlamingoLayer`
                    layer.condition_vis_x(vision)
        elif self.resampler_type == "depth_codebook":
            (
                vision_x, 
                vision_gripper, 
                depth_static, # 过codebook后的depth_static
                depth_gripper, # 过codebook后的depth_gripper
                loss_static, 
                perplexity_static, 
                loss_gripper, 
                perplexity_gripper,
                indices_static,
                indices_gripper,
                depth_static_before_codebook, depth_gripper_before_codebook,
            ) = self._encode_multi_sensor_with_vq(vision_x, vision_gripper, depth_static, depth_gripper)
        elif self.resampler_type == "depth_codebook_pred_depth":  # inference, 缺失深度模态
            assert kwargs.get("mode", "training") == "inference"
            if kwargs.get("mode", "training") == "inference":
                vision_x = self._encode_vision(vision_x)
                vision_gripper = self._encode_vision(vision_gripper)
                vision_x = self.perceiver(vision_x)
                vision_gripper = self.perceiver(vision_gripper)
                depth_static = self.depth_pred(vision_x.unsqueeze(dim=1))
                depth_gripper = self.depth_pred(vision_gripper.unsqueeze(dim=1))
                depth_static_before_codebook = depth_static
                depth_gripper_before_codebook = depth_gripper
                _, depth_static, _, _, indices_static = self.depth_vq(depth_static)
                _, depth_gripper, _, _, indices_gripper = self.depth_vq(depth_gripper)
                vision = []
                vision.append(vision_x)
                vision.append(vision_gripper)
                vision.append(depth_static)
                vision.append(depth_gripper)
                vision = torch.cat(vision, dim=2)  # (B * window_size, 1, 4 * num_latents, vision_embedding_dim + vision_embedding_dim)
                depth_pred_loss = 0.0
                for layer in self.lang_encoder._get_decoder_layers():
                    # the function `condition_vis_x` is defined in `models.flamingo_lm.FlamingoLayer`
                    layer.condition_vis_x(vision)
        elif self.resampler_type == "depth_codebook_ema":
            assert "mode" in kwargs.keys()
            (
                vision_x, 
                vision_gripper, 
                depth_static, # 过codebook后的depth_static
                depth_gripper, # 过codebook后的depth_gripper
                loss_static, 
                perplexity_static, 
                loss_gripper, 
                perplexity_gripper,
                indices_static,
                indices_gripper,
                depth_static_before_codebook, depth_gripper_before_codebook,
            ) = self._encode_multi_sensor_with_vq_ema(vision_x, vision_gripper, depth_static, depth_gripper, kwargs["mode"])
        elif self.resampler_type == "depth_codebook_ema_finetune":
            assert kwargs.get("mode", "training") == "inference", "depth_codebook_ema_finetune only support inference mode"
            (
                vision_x, 
                vision_gripper, 
                depth_static, # 过codebook后的depth_static
                depth_gripper, # 过codebook后的depth_gripper
                loss_static, 
                perplexity_static, 
                loss_gripper, 
                perplexity_gripper,
                indices_static,
                indices_gripper,
                depth_static_before_codebook, depth_gripper_before_codebook,
            ) = self._encode_multi_sensor_with_vq_ema(vision_x, vision_gripper, depth_static, depth_gripper, "inference")
        elif self.resampler_type == "depth_codebook_ema_pred_depth":
            assert kwargs.get("mode", "training") == "inference" or kwargs.get("mode", "training") == "qualitative", "depth_codebook_ema_pred_depth only support inference mode"
            if kwargs.get("mode", "training") == "inference":
                vision_x = self._encode_vision(vision_x)
                vision_gripper = self._encode_vision(vision_gripper)
                vision_x = self.perceiver(vision_x)
                vision_gripper = self.perceiver(vision_gripper)
                depth_static = self.depth_pred(vision_x.unsqueeze(dim=1))
                depth_gripper = self.depth_pred(vision_gripper.unsqueeze(dim=1))
                depth_static_before_codebook = depth_static
                depth_gripper_before_codebook = depth_gripper
                _, depth_static, _, _, indices_static = self.depth_vq(depth_static, training=False)
                _, depth_gripper, _, _, indices_gripper = self.depth_vq(depth_gripper, training=False)
                vision = []
                vision.append(vision_x)
                vision.append(vision_gripper)
                vision.append(depth_static)
                vision.append(depth_gripper)
                vision = torch.cat(vision, dim=2)  # (B * window_size, 1, 4 * num_latents, vision_embedding_dim + vision_embedding_dim)
                depth_pred_loss = 0.0
                for layer in self.lang_encoder._get_decoder_layers():
                    # the function `condition_vis_x` is defined in `models.flamingo_lm.FlamingoLayer`
                    layer.condition_vis_x(vision)
            elif kwargs.get("mode", "training") == "qualitative":
                vision_x = self._encode_vision(vision_x)
                vision_gripper = self._encode_vision(vision_gripper)
                vision_x = self.perceiver(vision_x)
                vision_gripper = self.perceiver(vision_gripper)
                depth_static_gt = self.depth_perceiver(self._encode_depth_sensor(depth_static))
                depth_gripper_gt = self.depth_perceiver(self._encode_depth_sensor(depth_gripper))
                depth_static = self.depth_pred(vision_x.unsqueeze(dim=1))
                depth_gripper = self.depth_pred(vision_gripper.unsqueeze(dim=1))
                before_error = torch.mean((depth_static_gt - depth_static)**2, dim=(1, 2, 3))
                # before_error += torch.nn.functional.mse_loss(depth_gripper_gt, depth_gripper, reduction="mean")
                depth_static_before_codebook = depth_static
                depth_gripper_before_codebook = depth_gripper
                _, depth_static_gt, _, _, indices_static_gt = self.depth_vq(depth_static_gt, training=False)
                _, depth_gripper_gt, _, _, indices_gripper_gt = self.depth_vq(depth_gripper_gt, training=False)
                _, depth_static, _, _, indices_static = self.depth_vq(depth_static, training=False)
                _, depth_gripper, _, _, indices_gripper = self.depth_vq(depth_gripper, training=False)
                after_error = torch.mean((depth_static_gt - depth_static)**2, dim=(1, 2, 3))
                # after_error += torch.nn.functional.mse_loss(depth_gripper_gt, depth_gripper, reduction="mean")
                from utils.global_var import set_value
                set_value("before_error", before_error)
                set_value("after_error", after_error)
                vision = []
                vision.append(vision_x)
                vision.append(vision_gripper)
                vision.append(depth_static)
                vision.append(depth_gripper)
                vision = torch.cat(vision, dim=2)
                depth_pred_loss = 0.0
                for layer in self.lang_encoder._get_decoder_layers():
                    # the function `condition_vis_x` is defined in `models.flamingo_lm.FlamingoLayer`
                    layer.condition_vis_x(vision)
        elif self.resampler_type == "KD":
            vision_x, vision_gripper, depth_static, depth_gripper = self._encode_multi_sensor_post_fusion(vision_x, vision_gripper, depth_static, depth_gripper)
            for layer in self.lang_encoder._get_decoder_layers():
                # the function `condition_vis_x` is defined in `models.flamingo_lm.FlamingoLayer`
                layer.condition_vis_x(torch.cat([vision_x, vision_gripper], dim=2))
        elif self.resampler_type == "MM_Prompt":
            vision_x, vision_gripper, depth_static, depth_gripper = self._encode_multi_sensor_post_fusion_with_prompt(vision_x, vision_gripper, depth_static, depth_gripper)
        else:
            raise NotImplementedError
        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask.bool(),
            output_hidden_states=True
        )
        # output.hidden_states = (
        #   torch.Tensor shape = (B * window_size, T_text, lang_embedding_dim),
        #   ...,
        # ) len(output.hidden_states) = len(self.lang_encoder.transformer.blocks)
        output_hs = output.hidden_states[-1]  # (B * window_size, T_text, lang_embedding_dim)
        output_hs = self.lm_head(output_hs)
        output.logits = output_hs  # (B, window_size, 6), (B, window_size, 1)
        if vision_x is not None:
            output.vision_x = vision_x.reshape(B, -1, *vision_x.shape[1:])  # (B, window_size, 1, num_latents, vision_embedding_dim)
            output.vision_gripper = vision_gripper.reshape(B, -1, *vision_gripper.shape[1:])  # (B, window_size, 1, num_latents, vision_embedding_dim)
        if depth_static is not None:
            output.depth_static = depth_static.reshape(B, -1, *depth_static.shape[1:])  # (B, window_size, 1, num_latents, depth_embedding_dim)
            output.depth_gripper = depth_gripper.reshape(B, -1, *depth_gripper.shape[1:])  # (B, window_size, 1, num_latents, depth_embedding_dim)
        elif self.resampler_type == "pred_depth":
            output.depth_pred_loss = depth_pred_loss
            output.depth_static_pred = depth_static_pred
            output.depth_gripper_pred = depth_gripper_pred
            # for visualize
            if kwargs.get("mode", "training") == "vis":
                output.last_attn = last_attn
                output.diff = diff
        elif self.resampler_type == "depth_codebook":
            output.loss_static = loss_static
            output.perplexity_static = perplexity_static
            output.loss_gripper = loss_gripper
            output.perplexity_gripper = perplexity_gripper
            output.indices_static = indices_static
            output.indices_gripper = indices_gripper
            output.depth_static_before_codebook = depth_static_before_codebook
            output.depth_gripper_before_codebook = depth_gripper_before_codebook
        elif self.resampler_type == "depth_codebook_pred_depth":
            output.indices_static = indices_static
            output.indices_gripper = indices_gripper
            output.depth_static_before_codebook = depth_static_before_codebook  # depth_pred预测的深度
            output.depth_gripper_before_codebook = depth_gripper_before_codebook  # depth_pred预测的深度
        elif self.resampler_type == "depth_codebook_ema":
            output.loss_static = loss_static
            output.perplexity_static = perplexity_static
            output.loss_gripper = loss_gripper
            output.perplexity_gripper = perplexity_gripper
            output.indices_static = indices_static
            output.indices_gripper = indices_gripper
            output.depth_static_before_codebook = depth_static_before_codebook
            output.depth_gripper_before_codebook = depth_gripper_before_codebook
        elif self.resampler_type == "depth_codebook_ema_finetune":
            output.loss_static = loss_static
            output.perplexity_static = perplexity_static
            output.loss_gripper = loss_gripper
            output.perplexity_gripper = perplexity_gripper
            output.indices_static = indices_static
            output.indices_gripper = indices_gripper
            output.depth_static_before_codebook = depth_static_before_codebook
            output.depth_gripper_before_codebook = depth_gripper_before_codebook
        elif self.resampler_type == "depth_codebook_ema_pred_depth":
            output.indices_static = indices_static
            output.indices_gripper = indices_gripper
            output.depth_static_before_codebook = depth_static_before_codebook
            output.depth_gripper_before_codebook = depth_gripper_before_codebook
        elif self.resampler_type == "KD":
            pass
        return output

def load_DepthMPTFlamingo(
    lang_encoder_path: str,
    vision_encoder_path: str = "ViT-L-14",
    vision_encoder_pretrained: str = "openai",
    vision_encoder_cache_dir: Optional[str] = None,
    decoder_layers_attr_name: str = "transformer.blocks",
    cross_attn_every_n_layers: int = 1,
    window_size: int = 12,
    freeze_lm: bool = False,
    freeze_rgb: bool = False,
    resampler_type: str = "perceiver_resampler",
    **kwargs
):
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        model_name=vision_encoder_path,
        pretrained=vision_encoder_pretrained,
        cache_dir=vision_encoder_cache_dir
    )
    vision_encoder.visual.output_tokens = True

    text_tokenizer = AutoTokenizer.from_pretrained(Path(lang_encoder_path))
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    lang_encoder = AutoModelForCausalLM.from_pretrained(
        Path(lang_encoder_path), 
        trust_remote_code=True
    )
    if "mpt-1b-redpajama-200b" in lang_encoder_path:
        class EmbeddingFnMixin:
            def get_input_embeddings(self):
                return self.transformer.wte

            def set_input_embeddings(self, new_embeddings):
                self.transformer.wte = new_embeddings
        extend_instance(lang_encoder, EmbeddingFnMixin)
    extend_instance(lang_encoder, FlamingoLMMixin)
    # the following function `set_decoder_layers_attr_name` is defined in `FlamingoLMMixin`
    lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    model = DepthMPTFlamingo(
        vision_encoder=vision_encoder,
        depth_sensor_encoder=vision_encoder,
        lang_encoder=lang_encoder,
        media_token_id=text_tokenizer.encode("<image>")[-1],
        vision_embedding_dim=open_clip.get_model_config(vision_encoder_path)["vision_cfg"]["width"],
        depth_embedding_dim=open_clip.get_model_config(vision_encoder_path)["vision_cfg"]["width"],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        window_size=window_size,
        resampler_type=resampler_type,
    )
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0, f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"

    if not freeze_lm:
        model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
        model.lang_encoder.lm_head.requires_grad_(True)  # This is a nn.Identity()
    else:
        print("Training with freezing LM")
    if not freeze_rgb:
        model.perceiver.requires_grad_(True)
    model.lm_head.requires_grad_(True)
    if resampler_type == "perceiver_resampler":
        model.depth_perceiver.requires_grad_(True)
    if resampler_type == "pred_depth":
        model.depth_perceiver.requires_grad_(True)
        model.depth_pred.requires_grad_(True)
    if resampler_type == "depth_codebook":
        model.depth_perceiver.requires_grad_(True)
        model.depth_vq.requires_grad_(True)
    if resampler_type == "depth_codebook_pred_depth":
        model.depth_perceiver.requires_grad_(True)
        model.depth_vq.requires_grad_(True)
        model.depth_pred.requires_grad_(True)
    if resampler_type == "depth_codebook_ema":
        model.depth_perceiver.requires_grad_(True)
        model.depth_vq.requires_grad_(True)
    if resampler_type == "depth_codebook_ema_finetune":
        model.depth_perceiver.requires_grad_(False)
        model.perceiver.requires_grad_(False)
        model.depth_vq.requires_grad_(False)
    if resampler_type == "KD":
        model.depth_perceiver.requires_grad_(False)
    if resampler_type == "MM_Prompt":
        model.depth_perceiver.requires_grad_(True)
        model.prompt.requires_grad_(True)
    if resampler_type == "pred_depth_no_latent":
        model.depth_perceiver.requires_grad_(True)
        model.depth_pred.requires_grad_(True)

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, text_tokenizer
