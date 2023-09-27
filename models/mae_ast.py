# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Dict, List, Optional, Tuple
import random
import numpy as np

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import utils
from fairseq.data.data_utils import compute_mask_indices
from fairseq.data.dictionary import Dictionary
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import BaseFairseqModel, register_model
from fairseq.models.wav2vec.wav2vec2 import (
    ConvFeatureExtractionModel,
    # TransformerEncoder,
)
# from fairseq.modules import (
#     SinusoidalPositionalEmbedding
# )
from fairseq.modules import GradMultiply, LayerNorm

from omegaconf import II

logger = logging.getLogger(__name__)

MASKING_DISTRIBUTION_CHOICES = ChoiceEnum(["static", "uniform", "normal", "poisson"])
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from dataclasses import dataclass, field
from fairseq.data import Dictionary
from mae_ast.data import MAE_AST_Dataset
from fairseq.dataclass import ChoiceEnum
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
from fairseq.tasks.fairseq_task import FairseqTask
from omegaconf import MISSING

logger = logging.getLogger(__name__)

MASK_TYPE_CHOICES = ChoiceEnum(["retain_spans", "random_mask", "random_mask_batched", "chunk_mask"])


@dataclass
class MAE_AST_Pretraining_Config(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})

    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down "
                    "sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False,
        metadata={"help": "pad shorter samples instead of cropping"},
    )
    max_keep_size: Optional[int] = field(
        default=None,
        metadata={"help": "exclude sample longer than this"},
    )
    max_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "max sample size to crop to for batching"},
    )
    min_sample_size: Optional[int] = field(
        default=None,
        metadata={"help": "min sample size to crop to for batching"},
    )
    random_crop: Optional[bool] = field(
        default=True,
        metadata={"help": "always crop from the beginning if false"},
    )
    pad_audio: Optional[bool] = field(
        default=False,
        metadata={"help": "pad audio to the longest one in the batch if true"},
    )

    feature_type: Optional[str] = field(
        default='wav',
        metadata={"help": "choose from ['wav', 'spectrogram', 'fbank', 'mfcc']"}
    )

    feature_rate: Optional[int] = field(
        default=100,
        metadata={
            "help": "rate of feature input to the transformer, if use wav, this arg is omited, else if use spectrogram/fbank/mfcc, the default is 100, i.e. 1s audio gives 100 frames. the label rate of using MFCC is also 100"}
    )

    feature_dim: Optional[int] = field(
        default=100,
        metadata={
            "help": "dim feature input to the transformer, if use wav, this arg is omited, else if use spectrogram/fbank/mfcc, the default is 80"}
    )

    deltas: Optional[bool] = field(
        default=True,
        metadata={
            "help": "whether or not add delta and delta-delta to the feature, only effective for spectrogram/fbank/mfcc"}
    )

    mask_spans: Optional[bool] = field(
        default=False,
        metadata={"help": "mask random spans, same as that is used in HuBERT and w2v2"}
    )

    mask_type: MASK_TYPE_CHOICES = field(
        default='random_mask',
        metadata={"help":
                      """Determine type of mask for MAE pretraining. 
                      -retain_spans: Only for frame data. Wav2Vec2 like masking.
                      -random_mask: Perform masking on completely random tokens. No chunking. Used in MAE
                      -random_mask_batched: random_mask with the same mask across the batch.
                      -chunk_mask: Perform masking on chunks until mask_spans hit. From SSAST. Same across batch for speed.
                          """}
    )


@register_task("mae_ast_pretraining", dataclass=MAE_AST_Pretraining_Config)
class MAE_AST_Pretraining_Task(FairseqTask):
    cfg: MAE_AST_Pretraining_Config

    def __init__(
            self,
            cfg: MAE_AST_Pretraining_Config,
    ) -> None:
        super().__init__(cfg)

        logger.info(f"current directory is {os.getcwd()}")
        logger.info(f"MAEPretrainingTask Config {cfg}")

        self.cfg = cfg

    @property
    def source_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def target_dictionary(self) -> Optional[Dictionary]:
        return None

    @property
    def dictionaries(self) -> List[Dictionary]:
        return None

    @classmethod
    def setup_task(
            cls, cfg: MAE_AST_Pretraining_Config, **kwargs
    ) -> "MAE_AST_Pretraining_Task":
        return cls(cfg)

    def load_dataset(self, split: str, **kwargs) -> None:
        manifest = f"{self.cfg.data}/{split}.tsv"

        self.datasets[split] = MAE_AST_Dataset(
            manifest,
            sample_rate=self.cfg.sample_rate,
            max_keep_sample_size=self.cfg.max_keep_size,
            min_keep_sample_size=self.cfg.min_sample_size,
            max_sample_size=self.cfg.max_sample_size,
            pad_audio=self.cfg.pad_audio,
            normalize=self.cfg.normalize,
            random_crop=self.cfg.random_crop,
            feature_type=self.cfg.feature_type,
            feature_dim=self.cfg.feature_dim,
            deltas=self.cfg.deltas,
            feature_rate=self.cfg.feature_rate
        )

    def max_positions(self) -> Tuple[int, int]:
        return (sys.maxsize, sys.maxsize)

    def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
        return indices

@dataclass
class MAE_AST_Config(FairseqDataclass):
    # Patching settings (frame vs patch based model)
    ast_kernel_size_chan: int = field(
        default=16,
        metadata={
            "help": "When using reconstruction on image data, this sets the kernel size for channels (size and stride must be identical for reconstruction) used for masked features. Default 16, see ast_models.py"}
    )
    ast_kernel_size_time: int = field(
        default=16,
        metadata={
            "help": "When using reconstruction on image data, this sets the kernel size for time (size and stride must be identical for reconstruction) used for masked features. Default 16, see ast_models.py."}
    )
    ast_kernel_stride_chan: int = field(
        default=16,
        metadata={
            "help": "When using reconstruction on image data, this sets the kernel stride for channels (size and stride must be identical for reconstruction) used for masked features. Default 16, see ast_models.py"}
    )
    ast_kernel_stride_time: int = field(
        default=16,
        metadata={
            "help": "When using reconstruction on image data, this sets the kernel stride for time (size and stride must be identical for reconstruction) used for masked features. Default 16, see ast_models.py."}
    )

    # Encoder and general transformer settings
    encoder_layers: int = field(
        default=12, metadata={"help": "num encoder layers in the encoder transformer"}
    )
    encoder_embed_dim: int = field(
        default=768, metadata={"help": "encoder embedding dimension"}
    )
    encoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "encoder embedding dimension for FFN"}
    )
    encoder_attention_heads: int = field(
        default=12, metadata={"help": "num encoder attention heads"}
    )
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="gelu", metadata={"help": "activation function to use"}
    )

    layer_norm_first: bool = field(
        default=False,
        metadata={"help": "apply layernorm first in the transformer"},
    )
    feature_grad_mult: float = field(
        default=1.0,
        metadata={"help": "multiply feature extractor var grads by this"},
    )

    # Decoder settings
    use_post_enc_proj: bool = field(
        default=False,
        metadata={
            "help": "Linear projection on the encoder output. Required if decoder embed dim != encoder embed dim"}
    )
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_layers: int = field(
        default=2, metadata={"help": "num encoder layers in the decoder transformer"}
    )
    decoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a decoder transformer layer. The decoder is shallow, so this should typically be 0"},
    )

    # Dropouts
    dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for the transformer"},
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "dropout probability for attention weights"},
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability after activation in FFN"},
    )
    encoder_layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a transformer layer"},
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )

    # Overall Masking Settings
    random_mask_prob: float = field(
        default=0.75,
        metadata={"help": "Probability of a given token being masked. Exact use depends on mask type"}
    )

    # Wav2Vec2-like Masking settings
    mask_length: int = field(default=10, metadata={"help": "mask length"})

    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose mask length"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
                    "(used for more complex distributions), "
                    "see help in compute_mask_indicesh"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: int = field(
        default=0,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )

    # Convolutional positional embeddings (not used for MAE-AST)
    conv_pos: int = field(
        default=128,
        metadata={"help": "number of filters for convolutional positional embeddings"},
    )
    conv_pos_groups: int = field(
        default=16,
        metadata={"help": "number of groups for convolutional positional embedding"},
    )

    # loss computation
    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "recompute activations and save memory for extra compute"},
    )

    # positional embeddings
    max_token_length: int = field(
        default=48000,
        metadata={"help": "the longest input sequence length, used for sinusoidal positional embedding"}
    )
    enc_sine_pos: bool = field(
        default=False,
        metadata={"help": "sinusoidal positional embeddings for encoder input"}
    )
    enc_conv_pos: bool = field(
        default=False,
        metadata={"help": "convnet positional embeddings for encoder input"}
    )
    dec_sine_pos: bool = field(
        default=False,
        metadata={"help": "sinusoidal positional embeddings for decoder input"}
    )
    dec_conv_pos: bool = field(
        default=False,
        metadata={"help": "convnet positional embeddings for decoder input"}
    )




"""
    MAE-AST (MAE-AST: Masked Autoencoding Audio Spectrogram Transformer)
    https://github.com/AlanBaade/MAE-AST-Public

    MIT license
"""

# MAE-AST:

from fairseq.models import BaseFairseqModel


class MAE_AST(BaseFairseqModel):
    def __init__(
            self,
            cfg: MAE_AST_Config,
            task_cfg: MAE_AST_Pretraining_Task,
    ) -> None:
        super().__init__()
        logger.info(f"MAEModel Config: {cfg}")
        self.cfg = cfg
        self.task_cfg = task_cfg

        self.feature_extractor = nn.Identity()
        self.post_extract_proj = nn.Linear(cfg.ast_kernel_size_time * cfg.ast_kernel_size_chan, cfg.encoder_embed_dim)
        self.layer_norm = LayerNorm(task_cfg.feature_dim)
        self.batch_norm = nn.BatchNorm2d(num_features=1, affine=False)
        self.unfold = nn.Unfold(kernel_size=(cfg.ast_kernel_size_time, cfg.ast_kernel_size_chan),
                                stride=(cfg.ast_kernel_stride_time, cfg.ast_kernel_stride_chan))

        self.is_batched_mask = self.task_cfg.mask_type == 'random_mask_batched' or self.task_cfg.mask_type == 'chunk_patch'

        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)

        self.feature_grad_mult = cfg.feature_grad_mult

        self.encoder_mask_emb = nn.Parameter(torch.FloatTensor(cfg.encoder_embed_dim).uniform_())

        if self.cfg.enc_conv_pos:
            self.enc_conv_pos_embed = ConvPosEmbed(cfg)
        if self.cfg.enc_sine_pos:
            self.enc_sine_pos_embed = SinusoidalPositionalEncoding(d_model=self.cfg.encoder_embed_dim,
                                                                   max_len=self.cfg.max_token_length)

        self.encoder = TransformerEncoder(cfg)

        if self.cfg.use_post_enc_proj:
            self.post_enc_proj = nn.Linear(self.cfg.encoder_embed_dim, self.cfg.decoder_embed_dim)
        else:
            assert self.cfg.decoder_embed_dim == self.cfg.encoder_embed_dim, "Need post_enc_projection if encoder and decoder embed dims differ"

        self.decoder_mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.decoder_embed_dim).uniform_())
        if self.cfg.dec_conv_pos:
            self.dec_conv_pos_embed = ConvPosEmbed(cfg)
        if self.cfg.dec_sine_pos:
            self.dec_sine_pos_embed = SinusoidalPositionalEncoding(d_model=self.cfg.decoder_embed_dim,
                                                                   max_len=self.cfg.max_token_length)
        # decoder uses the same set of params as the encoders, except for the number of layers and width
        self.decoder = TransformerEncoder(cfg, decoder=True)

        self.final_proj_reconstruction = nn.Linear(cfg.decoder_embed_dim,
                                                   cfg.ast_kernel_size_time * cfg.ast_kernel_size_chan)
        self.final_proj_classification = nn.Linear(cfg.decoder_embed_dim,
                                                   cfg.ast_kernel_size_time * cfg.ast_kernel_size_chan)

        assert self.task_cfg.feature_type != "wav"

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: MAE_AST_Config, task: MAE_AST_Pretraining_Task):
        """Build a new model instance."""

        model = MAE_AST(cfg, task.cfg)
        return model

    def forward_features(self, source: torch.Tensor) -> torch.Tensor:
        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)
        return features

    def forward_padding_mask(
            self,
            features: torch.Tensor,
            padding_mask: torch.Tensor,
            feature_dim=128,
    ) -> torch.Tensor:
        if (padding_mask[:, -1].sum() == 0):  # Fast exit during training if not necessary
            return padding_mask.new_zeros(features.shape[:2])

        non_zero_count = padding_mask.size(-1) - padding_mask.sum(dim=-1)
        num_patches_over_channel = feature_dim // self.cfg.ast_kernel_size_chan
        padding_mask_indices = (((non_zero_count - 1) // self.cfg.ast_kernel_size_time) + 1) * num_patches_over_channel
        new_padding_mask = padding_mask.new_zeros(features.shape[:2])

        for i in range(new_padding_mask.size(0)):
            if padding_mask_indices[i] < new_padding_mask.size(-1):
                new_padding_mask[i, padding_mask_indices[i]:] = True

        return new_padding_mask

    def forward_mask(self, features, padding_mask):
        B, T, C = features.shape
        num_retained_tokens = int((1 - self.cfg.random_mask_prob) * T)
        num_retained_tokens = max(1, num_retained_tokens)
        retained_idx = []
        masked_idx = []

        if self.task_cfg.mask_type == 'retain_spans':
            num_retained_tokens = 0
            while num_retained_tokens == 0:  # This loop will almost never run more than once for any reasonable mask ratio.
                retained_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.cfg.random_mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                )
                num_retained_tokens = retained_indices[0].sum()
            for i in range(B):
                cur_retained_idx = np.where(retained_indices[i])[0]
                cur_masked_idx = np.where(~retained_indices[i])[0]
                retained_idx.append(cur_retained_idx)
                features[i, cur_masked_idx] = self.encoder_mask_emb
                masked_idx.append(cur_masked_idx)
        elif self.task_cfg.mask_type == 'random_mask':
            for i in range(B):
                idx = list(range(T))
                random.shuffle(idx)
                cur_retained_idx = idx[:num_retained_tokens]
                retained_idx.append(cur_retained_idx)
                cur_masked_idx = idx[num_retained_tokens:]
                masked_idx.append(cur_masked_idx)
                features[i, cur_masked_idx] = self.encoder_mask_emb
        elif self.task_cfg.mask_type == 'random_mask_batched':
            idx = list(range(T))
            random.shuffle(idx)
            cur_retained_idx = idx[:num_retained_tokens]
            retained_idx = [cur_retained_idx]
            cur_masked_idx = idx[num_retained_tokens:]
            masked_idx = [cur_masked_idx]
            features[:, cur_masked_idx] = self.encoder_mask_emb
        elif self.task_cfg.mask_type == 'chunk_mask':  # Copies SSAST code and goes to bottom right from uniform starting index
            cur_masked_idx = set()
            chunk_size = random.randrange(3, 5 + 1)
            chan_adjust = self.task_cfg.feature_dim // self.cfg.ast_kernel_stride_chan
            num_masked_tokens = T - num_retained_tokens
            while len(cur_masked_idx) <= num_masked_tokens:
                t_topleft = random.randrange(T)
                for t_offset in range(0, chunk_size):
                    for c_offset in range(0, chunk_size):
                        mask_cand = t_topleft + t_offset + chan_adjust * c_offset
                        if (mask_cand < T):
                            cur_masked_idx.add(mask_cand)
            cur_masked_idx = list(cur_masked_idx)
            cur_masked_idx = cur_masked_idx[:num_masked_tokens]
            cur_retained_idx = list(set(range(T)).difference(cur_masked_idx))
            for i in range(B):  # Using same mask for whole batch because SSAST code is very slow
                retained_idx.append(cur_retained_idx)
                masked_idx.append(cur_masked_idx)
                features[i, cur_masked_idx] = self.encoder_mask_emb

        return retained_idx, masked_idx, T - num_retained_tokens

    def forward(
            self,
            source: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
            mask: bool = True,
            features_only: bool = False,
            output_layer: Optional[int] = None,
            is_decoder_finetune: bool = False,
            is_input_prepatched: bool = False,
    ) -> Dict[str, torch.Tensor]:

        # Checks whether the dataset was patched and normalized before-hand. is_input_prepatched == True for speed profiling during training.
        if is_input_prepatched:
            source_patch = source
        else:
            # Batch normalization ('Mimics' AST dataset normalization)
            source = source.unsqueeze(1)
            source = self.batch_norm(source) * 0.5  # Mean 0, St dev 0.5
            # Create image patches for masking via Unfold. BTC input shape BTC output shape
            # Output continues to be unsqueezed from batch norm
            source_patch = self.unfold(source).transpose(-1, -2)

        features = self.forward_features(source_patch)
        if padding_mask is not None:  # Reshape padding mask
            padding_mask = self.forward_padding_mask(features, padding_mask)
        if self.post_extract_proj is not None:  # Project patches to vectors of size encoder_dim
            features = self.post_extract_proj(features)
        if mask:  # additional regularization adopted from hubert
            features = self.dropout_input(features)

        B, T, C = features.shape

        # Calculate retained_idx and masked_idx. Uses safe assumption that nothing is padded during pretraining
        if mask:
            retained_idx, masked_idx, num_masked_tokens = self.forward_mask(features, padding_mask)
        else:
            retained_idx = []
            masked_idx = []
            num_masked_tokens = 0

        # Pre-Encoder Positional Embeddings
        if self.cfg.enc_conv_pos:
            conv_pos = self.enc_conv_pos_embed(features, padding_mask)
            features = conv_pos + features
        if self.cfg.enc_sine_pos:
            sine_pos = self.enc_sine_pos_embed(features, padding_mask)
            features = sine_pos + features

        # Remove masked tokens from features
        if mask:
            if self.is_batched_mask:
                x = features[:, retained_idx[0]]
                retained_padding_mask = padding_mask[:, retained_idx[0]]
            else:
                x = []
                retained_padding_mask = []
                for i in range(B):
                    x.append(features[i, retained_idx[i]])
                    retained_padding_mask.append(padding_mask[i, retained_idx[i]])
                x = torch.stack(x, dim=0)
                retained_padding_mask = torch.stack(retained_padding_mask, dim=0)
        else:
            x = features
            retained_padding_mask = padding_mask

        # Encoder forward pass + Early return for features
        x, encoder_hidden_states = self.encoder(
            x,
            padding_mask=retained_padding_mask,
            layer=None if output_layer is None else output_layer - 1,
        )

        if not is_decoder_finetune and (features_only or not mask):
            return {"x": x, "padding_mask": retained_padding_mask, "features": features,
                    "hidden_states": encoder_hidden_states}

        if self.cfg.use_post_enc_proj:
            x = self.post_enc_proj(x)

        # Add masked tokens back
        if mask:
            full_x = torch.empty((B, T, C), device=x.device, dtype=x.dtype)
            mask_indices = torch.zeros(torch.Size([B, T]), device=padding_mask.device, dtype=torch.bool)
            if self.is_batched_mask:
                full_x[:, retained_idx[0]] = x
                full_x[:, masked_idx[0]] = self.decoder_mask_emb
                mask_indices[:, masked_idx[0]] = True
            else:
                for i, (cur_feat, ridx, midx) in enumerate(zip(x, retained_idx, masked_idx)):
                    full_x[i, ridx] = cur_feat
                    full_x[i, midx] = self.decoder_mask_emb
                    mask_indices[i, midx] = True
        else:
            full_x = x

        # Pre decoder positional embeddings
        if self.cfg.dec_conv_pos:
            conv_pos = self.dec_conv_pos_embed(full_x, padding_mask)
            full_x = conv_pos + full_x
        if self.cfg.dec_sine_pos:
            # Concerning that magnitudes of layer-normed encoder outputs are similar to decoder positional embeddings.
            sine_pos = self.dec_sine_pos_embed(full_x, padding_mask)
            full_x = sine_pos + full_x

        # Decoder forward pass
        x, decoder_hidden_states = self.decoder(full_x, padding_mask=padding_mask, layer=None)

        if is_decoder_finetune:
            return {"x": x, "padding_mask": padding_mask, "features": features,
                    "hidden_states": encoder_hidden_states + decoder_hidden_states}

        # Construct linear projection logits and masked reconstruction targets
        x_masked_indices = x[mask_indices].view(B, num_masked_tokens, -1)

        logit_m_list_recon = self.final_proj_reconstruction(x_masked_indices)
        logit_m_list_class = self.final_proj_classification(x_masked_indices)

        target_m_list = source_patch[mask_indices].view(B, num_masked_tokens, -1)

        result = {
            "logit_m_list_recon": logit_m_list_recon,
            "logit_m_list_class": logit_m_list_class,
            "target_m_list": target_m_list,
            "padding_mask": padding_mask,
        }
        return result

    def extract_features(
            self,
            source: torch.Tensor,
            padding_mask: Optional[torch.Tensor] = None,
            mask: bool = False,
            ret_conv: bool = False,
            output_layer: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            output_layer=output_layer,
        )
        feature = res["features"] if ret_conv else res["x"]
        return feature, res["padding_mask"]

    def get_logits(self, net_output):
        logits_list = net_output["logit_m_list_recon"], net_output["logit_m_list_class"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        return net_output["target_m_list"].float()

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.final_proj_reconstruction = None
        self.final_proj_classification = None