"""
Origin Repo: https://github.com/RoboFlamingo/RoboFlamingo
"""

import copy
from pathlib import Path
from typing import Optional
from collections import namedtuple

import torch
import open_clip
import torch.nn as nn
from einops import rearrange, repeat
from transformers import AutoTokenizer, AutoModelForCausalLM

from models.roboflamingo.helpers import PerceiverResampler
from models.roboflamingo.flamingo_lm import FlamingoLMMixin
from models.roboflamingo.flamingo_utils import extend_instance

def lstm_decoder(in_features: int, hidden_size: int, num_layers: int, policy_rnn_dropout_p: float) -> torch.nn.Module:
    return nn.LSTM(
        input_size=in_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=False,
        batch_first=True,
        dropout=policy_rnn_dropout_p,
    )

class MLPTanhHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
            torch.nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x)

class MLPSigmoidHead(torch.nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, output_size),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x)

class DeterministicDecoder(nn.Module):
    
    def __init__(
        self,
        in_features: int,
        window_size: int,
        hidden_size: int = 1024,
        num_layers: int = 4,
        policy_rnn_dropout_p: float = 0.1,
        # use_diff=False,
        # last_action=False,
        # fusion_mode='',
        # use_state=False,
        # multi_step_action=1,
        # return_feature=False,
        # pooling="max",
    ) -> None:
        super().__init__()
        self.fc_state = None
        self.in_features = in_features
        self.window_size = window_size

        self.history_len = window_size
        self.history_memory = []
        
        self.rnn = lstm_decoder(in_features, hidden_size, num_layers, policy_rnn_dropout_p)
        self.actions = MLPTanhHead(hidden_size, 6)
        self.gripper = MLPSigmoidHead(hidden_size, 1)
        
        self.hidden_state = None
        self.hidden_size = hidden_size
        self.rnn_out = None
        self.last_action = False
        self.global_1d_pool = nn.AdaptiveMaxPool1d(1)
    
    def clear_hidden_state(self) -> None:
        self.hidden_state = None

    def forward(
        self, 
        input_feature: torch.Tensor,
        h_0: Optional[torch.Tensor] = None,
        *args,
        **kwargs
    ):
        """
        Args:
            input_feature (torch.Tensor): (B * window_size, T_text, lang_embedding_dim)
            h_0 (Optional[torch.Tensor], optional): [description]. Defaults to None.
        Returns:
            actions (torch.Tensor): (B, window_size, 6)
            gripper (torch.Tensor): (B, window_size, 1)
        """
        if input_feature.dim() == 3:  # (B * window_size, T_text, lang_embedding_dim)
            input_feature = self.global_1d_pool(input_feature.permute(0, 2, 1)).squeeze(-1)  # (B * window_size, lang_embedding_dim)
        input_feature = input_feature.reshape(-1, self.window_size, input_feature.shape[1])  # (B, window_size, lang_embedding_dim)
        
        if input_feature.shape[1] == 1:
            self.history_memory.append(input_feature)
            if len(self.history_memory) <= self.history_len:
                x, h_n = self.rnn(input_feature, self.hidden_state)
                self.hidden_state = h_n
                x = x[:, -1].unsqueeze(1)
                self.rnn_out = x.squeeze(1)
            else:
                cur_len = len(self.history_memory)
                for _ in range(cur_len - self.history_len):
                    self.history_memory.pop(0)
                assert len(self.history_memory) == self.history_len
                hist_feature = torch.cat(self.history_memory, dim=1)
                self.hidden_state = None
                x, h_n = self.rnn(hist_feature, self.hidden_state)
                x = x[:, -1].unsqueeze(1)
                self.rnn_out = x.squeeze(1)
        else:
            self.hidden_state = h_0
            x, h_n = self.rnn(input_feature, self.hidden_state)
            self.hidden_state = h_n
            if self.last_action:
                x = x[:, -1].unsqueeze(1)
            self.rnn_out = x.squeeze(1)
        actions = self.actions(x)  # (B, window_size, 6)
        gripper = self.gripper(x)  # (B, window_size, 1)
        return actions, gripper

class MPTFlamingo(nn.Module):

    def __init__(
        self,
        vision_encoder: nn.Module,
        lang_encoder: nn.Module,
        media_token_id: int,  # token id for <image>
        vision_embedding_dim: int,  # 1024
        cross_attn_every_n_layers: int, # 1
        # use_media_placement_augmentation: bool = False,
        window_size: int,  # 12
        # use_gripper: bool, # True
        # fusion_mode: str,  # post
        # sep_resampler: bool, # False 是否使用不同的perceiver resampler层
        # use_state: bool,  # False, 是否使用state信息
        # last_action: bool, # False, 是否使用last action信息
        # n_timestamps: int,  # 150, 只在diffusion模式下使用
        # state_dim: int,  # 15
        # use_hist: bool,  # False
        # predict_epsilon: bool # False, 只在diffusion模式下使用
        # pad_length: int  # -1, 没找到用处
        # multi_step_action: int  # 1, 预测多步动作
        # sep_lm_head: bool,  # 默认被设置为True
        # return_feature: bool,  # False, 是否返回特征
        # llm: str, # "llama", 只使用MPT模型
        # pooling: str, # max
        # residual: bool, # False, 只在llama中使用
        # tcp_rel: bool, # False, 没找到用处
        # replan: int, # -1, 没找到用处
        # decoder_type: str, # lstm, HF上的checkpoint只有lstm
        # hidden_size: int, # MPT模型使用lang_encoder.config.d_model
        # fwd_pred: bool, # False, 没找到用处
        # fwd_pred_hand: bool, # False, 没找到用处
        # global_latent: bool, # False, 没找到用处
        # no_image_patch: bool, # False, 没找到用处
        # refresh: int, # -1, 没找到用处
    ) -> None:
        """
        
        Args:
            vision_encoder (nnModule): HF CLIPModel
            lang_encoder_name (str): HF model name
            lang_encoder (nnModule): HF causal language model
            media_token_id (int): Token id for <image>
            vision_embedding_dim (int): Dimension of the visual features 1024
            cross_attn_every_n_layers (int):  How often to apply cross attention after transformer layer. Defaults to 1.
            window_size (int): 
            use_state (bool): Whether to use state information. Defaults to False.
            state_dim (int): Dimension of the state information. Defaults to 15.
        """
        super().__init__()
        self.vision_encoder = vision_encoder
        self.vision_embedding_dim = vision_embedding_dim

        self.lang_encoder = lang_encoder
        self.lang_embedding_dim = lang_encoder.config.d_model

        self.perceiver = PerceiverResampler(dim=self.vision_embedding_dim)

        self.media_token_id = media_token_id
        self.cross_attn_every_n_layers = cross_attn_every_n_layers
        self.window_size = window_size
        # self.use_state = use_state
        # self.state_dim = state_dim

        self.lang_encoder.init_flamingo(
            media_token_id=media_token_id,
            lang_hidden_size=self.lang_embedding_dim,
            vis_hidden_size=self.vision_embedding_dim,
            cross_attn_every_n_layers=self.cross_attn_every_n_layers,
            gradient_checkpointing=False,
        )
        
        # if use_state:
        #     self.state_fc = nn.Linear(state_dim, self.vision_embedding_dim)

        self.lang_encoder.lm_head = DeterministicDecoder(
            in_features=self.lang_embedding_dim,
            window_size=self.window_size,
        )
        self.lm_head = self.lang_encoder.lm_head
        self.lang_encoder.lm_head = nn.Identity()

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

    def _encode_multi_vision_post_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor):
        """
        
        Args:
            vision_rgb (torch.Tensor): shape = (B * window_size, 1, 1, 3, H, W)
            vision_gripper (torch.Tensor): shape = (B * window_size, 1, 1, 3, H, W)
        """
        vision_rgb = self._encode_vision(vision_rgb)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
        vision_gripper = self._encode_vision(vision_gripper)  # (B * window_size, 1, 1, patch_num, vision_embedding_dim)
        vision_rgb = self.perceiver(vision_rgb)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
        vision_gripper = self.perceiver(vision_gripper)  # (B * window_size, 1, num_latents, vision_embedding_dim), num_latents = 64
        vision_x = torch.cat([vision_rgb, vision_gripper], dim=2)  # (B * window_size, 1, 2 * num_latents, vision_embedding_dim), num_latents = 64
        for layer in self.lang_encoder._get_decoder_layers():
            # the function `condition_vis_x` is defined in `models.flamingo_lm.FlamingoLayer`
            layer.condition_vis_x(vision_x)

    def forward(
        self, 
        vision_x: torch.Tensor,
        lang_x: torch.Tensor,
        attention_mask: torch.Tensor,
        vision_gripper: torch.Tensor,
        *args,
        **kwargs
    ):
        """
        Args:
            vision_x (torch.Tensor): shape = (B, window_size, 1, 1, 3, H, W)
            lang_x (torch.Tensor): shape = (B, window_size, T_text)
            attention_mask (torch.Tensor): shape = (B, window_size, T_text)
            vision_gripper (torch.Tensor): shape = (B, window_size, 1, 1, 3, H, W)
        Returns:

        """
        B = vision_x.shape[0]
        vision_x = vision_x.reshape(-1, *vision_x.shape[2:])
        lang_x = lang_x.reshape(-1, *lang_x.shape[2:])
        attention_mask = attention_mask.reshape(-1, *attention_mask.shape[2:])
        vision_gripper = vision_gripper.reshape(-1, *vision_gripper.shape[2:])
        self._encode_multi_vision_post_fusion(vision_x, vision_gripper)
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
        return output
    
def load_MPTFlamingo(
    lang_encoder_path: str,
    vision_encoder_path: str = "ViT-L-14",
    vision_encoder_pretrained: str = "openai",
    vision_encoder_cache_dir: Optional[str] = None,
    decoder_layers_attr_name: str = "transformer.blocks",
    cross_attn_every_n_layers: int = 1,
    window_size: int = 12,
    freeze_lm: bool = False,
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

    model = MPTFlamingo(
        vision_encoder=vision_encoder,
        lang_encoder=lang_encoder,
        media_token_id=text_tokenizer.encode("<image>")[-1],
        vision_embedding_dim=open_clip.get_model_config(vision_encoder_path)["vision_cfg"]["width"],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        window_size=window_size
    )
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0, f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}"

    if not freeze_lm:
        model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
        model.lang_encoder.lm_head.requires_grad_(True)  # This is a nn.Identity()
    else:
        print("Training with freezing LM")
    model.lm_head.requires_grad_(True)
    model.perceiver.requires_grad_(True)

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, text_tokenizer
