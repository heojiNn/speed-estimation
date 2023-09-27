import os
import math
import wget
import warnings
import random
from random import randrange
import collections.abc
from itertools import repeat
import audtorch
import torchaudio
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torch.utils.checkpoint as checkpoint
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import audobject

import timm
from timm.models.layers import to_2tuple, trunc_normal_

# PaSST:
from hear21passt.base import get_basic_model

# HTS-AT:
import models.config_hts as config

# project dependant imports:
from models.util import do_mixup2 as do_mixup, interpolate
from tools.augmentation import FilterAugment

os.environ['TORCH_HOME'] = '../../pretrained_models'

########################################################################################################################
""" 
    AST (Audio Spectogram Transformer): https://github.com/YuanGongND/ast 
    & 
    SSAST: (Self-Supervised Audio Spectrogram Transformer) https://github.com/YuanGongND/ssast
    
    BSD 3-Clause License
"""


# AST:

# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
        """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ASTModel(nn.Module, audobject.Object):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """

    def __init__(
            self,
            label_dim=527,  # num_classes
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=1024,
            imagenet_pretrain=True,
            audioset_pretrain=False,
            model_size='base384',
            verbose=True):

        super(ASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        self.label_dim = label_dim,
        self.fstride = fstride,
        self.tstride = tstride,
        self.input_fdim = input_fdim,
        self.input_tdim = input_tdim,
        self.imagenet_pretrain = imagenet_pretrain,
        self.audioset_pretrain = audioset_pretrain,
        self.model_size = model_size,
        self.verbose = verbose

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),
                                                                                  str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim)
            )

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches,
                                                                            self.original_embedding_dim).transpose(1,
                                                                                                                   2).reshape(
                    1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :,
                                    int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(
                                        t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim),
                                                                    mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :,
                                    int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(
                                        f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(
                    torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError(
                    'currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists('models/checkpoints/audioset_10_10_0.4593.pth') == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='../../pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('models/checkpoints/audioset_10_10_0.4593.pth', map_location=device)
            audio_model = ASTModel(label_dim=527,
                                   fstride=10,
                                   tstride=10,
                                   input_fdim=128,
                                   input_tdim=1024,
                                   imagenet_pretrain=False,
                                   audioset_pretrain=False,
                                   model_size='base384',
                                   verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                          nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768,
                                                                                                              12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim / 2): 50 - int(t_dim / 2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim / 2): 6 - int(f_dim / 2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    @autocast()
    def forward(self, x):
        """
        :param x: the input spectrogram, expected shape: (batch_size, time_frame_num, frequency_bins), e.g., (16, 512, 64)
        :return: prediction
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        B = x.shape[0]  # Batchsize
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        x = self.mlp_head(x)
        return x


# SSAST (= AST + enhanced):

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class SSASTModel(nn.Module):
    def __init__(
            self,
            label_dim=527,
            fshape=128,
            tshape=2,
            fstride=128,
            tstride=2,
            input_fdim=128,
            input_tdim=1024,
            model_size='base',
            pretrain_stage=True,
            load_pretrained_mdl_path=None
    ):
        super(SSASTModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        self.label_dim = label_dim,
        self.fshape = fshape
        self.tshape = tshape
        self.fstride = fstride,
        self.tstride = tstride,
        self.input_fdim = input_fdim,
        self.input_tdim = input_tdim,
        self.model_size = model_size,
        self.pretrain_stage = pretrain_stage,
        self.load_pretrained_mdl_path = load_pretrained_mdl_path

        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # pretrain the AST models
        if pretrain_stage:
            if load_pretrained_mdl_path is not None:
                raise ValueError(
                    'Setting load_pretrained_mdl_path at pretraining stage is useless, pretraining is always from scratch, please change it to None.')
            if fstride != fshape or tstride != tshape:
                raise ValueError(
                    'fstride != fshape or tstride != tshape, they must be same at the pretraining stage, patch split overlapping is not supported.')

            # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
            if model_size == 'tiny':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 3, 12
                self.cls_token_num = 2
            elif model_size == 'small':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=False)
                self.heads, self.depth = 6, 12
                self.cls_token_num = 2
            elif model_size == 'base':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 2
            elif model_size == 'base_nokd':
                self.v = timm.create_model('vit_deit_base_patch16_384', pretrained=False)
                self.heads, self.depth = 12, 12
                self.cls_token_num = 1
            else:
                raise Exception('Model size must be one of tiny, small, base, base_nokd')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # SSL Pretraining Code
            self.softmax = nn.Softmax(dim=-1)
            self.lsoftmax = nn.LogSoftmax(dim=-1)
            self.fshape, self.tshape = fshape, tshape
            self.fstride, self.tstride = fstride, tstride
            self.input_fdim, self.input_tdim = input_fdim, input_tdim
            # this is a trick to make state_dict to track pretraining input_fdim and input_tdim and save them by using torch.save
            self.p_input_fdim, self.p_input_tdim = nn.Parameter(torch.tensor(input_fdim),
                                                                requires_grad=False), nn.Parameter(
                torch.tensor(input_tdim), requires_grad=False)

            # masked patch classification (discriminative objective) layer
            # we use two layers for pretext task, but using a single layer has similar performance.
            # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            self.cpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim),
                                            nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            # masked patch reconstruction (generative objective) layer
            self.gpredlayer = nn.Sequential(nn.Linear(self.original_embedding_dim, self.original_embedding_dim),
                                            nn.ReLU(), nn.Linear(self.original_embedding_dim, 256))
            self.unfold = torch.nn.Unfold(kernel_size=(fshape, tshape), stride=(fstride, tstride))

            # we use learnable mask embedding (follow the BEIT paper), but using a fixed mask embedding (e.g., 0) leads to same performance.
            self.mask_embed = nn.Parameter(torch.zeros([1, 1, self.original_embedding_dim]))
            self.mask_embed = torch.nn.init.xavier_normal_(self.mask_embed)

            # get the intermediate shape
            self.p_f_dim, self.p_t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            num_patches = self.p_f_dim * self.p_t_dim
            self.num_patches = num_patches
            self.v.patch_embed.num_patches = num_patches
            print('pretraining patch split stride: frequency={:d}, time={:d}'.format(fstride, tstride))
            print('pretraining patch shape: frequency={:d}, time={:d}'.format(fshape, tshape))
            print('pretraining patch array dimension: frequency={:d}, time={:d}'.format(self.p_f_dim, self.p_t_dim))
            print('pretraining number of patches={:d}'.format(num_patches))

            # the linear patch projection layer, use 1 channel for spectrogram rather than the original 3 channels for RGB images.
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape),
                                       stride=(fstride, tstride))
            self.v.patch_embed.proj = new_proj

            # use trainable positional embedding
            new_pos_embed = nn.Parameter(
                torch.zeros(1, self.v.patch_embed.num_patches + self.cls_token_num, self.original_embedding_dim))
            self.v.pos_embed = new_pos_embed
            trunc_normal_(self.v.pos_embed, std=.02)

        # use a pretrained models for finetuning
        elif pretrain_stage == False:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if load_pretrained_mdl_path == None:
                raise ValueError('Please set load_pretrained_mdl_path to load a pretrained models.')
            sd = torch.load(load_pretrained_mdl_path, map_location=device)
            # get the fshape and tshape, input_fdim and input_tdim in the pretraining stage
            try:
                p_fshape, p_tshape = sd['module.v.patch_embed.proj.weight'].shape[2], \
                                     sd['module.v.patch_embed.proj.weight'].shape[3]
                p_input_fdim, p_input_tdim = sd['module.p_input_fdim'].item(), sd['module.p_input_tdim'].item()
            except:
                raise ValueError(
                    'The model loaded is not from a torch.nn.Dataparallel object. Wrap it with torch.nn.Dataparallel and try again.')

            print('now load a SSL pretrained models from ' + load_pretrained_mdl_path)
            # during pretraining, fstride=fshape and tstride=tshape because no patch overlapping is used
            # here, input_fdim and input_tdim should be that used in pretraining, not that in the fine-tuning.
            # we need to know input_fdim and input_tdim to do positional embedding cut/interpolation.
            # generally it should be better to use same input_fdim during pretraining and finetuning, but input_tdim can be safely different
            audio_model = SSASTModel(fstride=p_fshape, tstride=p_tshape, fshape=p_fshape, tshape=p_tshape,
                                     input_fdim=p_input_fdim, input_tdim=p_input_tdim, pretrain_stage=True,
                                     model_size=model_size)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)

            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.cls_token_num = audio_model.module.cls_token_num

            # mlp head for fine-tuning
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim),
                                          nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape)
            # patch array dimension during pretraining
            p_f_dim, p_t_dim = audio_model.module.p_f_dim, audio_model.module.p_t_dim
            num_patches = f_dim * t_dim
            p_num_patches = p_f_dim * p_t_dim
            self.v.patch_embed.num_patches = num_patches
            print('fine-tuning patch split stride: frequncey={:d}, time={:d}'.format(fstride, tstride))
            print('fine-tuning number of patches={:d}'.format(num_patches))

            # patch shape should be same for pretraining and fine-tuning
            if fshape != p_fshape or tshape != p_tshape:
                raise ValueError(
                    'The patch shape of pretraining and fine-tuning is not consistant, pretraining: f={:d}, t={:d}, finetuning: f={:d}, t={:d}'.format(
                        p_fshape, p_tshape, fshape, tshape))

            # patch split stride generally should be different for pretraining and fine-tuning, as patch split overlapping is only used in finetuning
            # during pretraining, p_fshape = p_fstride and p_tshape = p_tstride
            if fstride != p_fshape or tstride != p_tshape:
                # initialize a new patch embedding layer with desired new stride.
                new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape),
                                           stride=(fstride, tstride))
                # but the weights of patch embedding layer is still got from the pretrained models
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
                self.v.patch_embed.proj = new_proj

            new_pos_embed = self.v.pos_embed[:, self.cls_token_num:, :].detach().reshape(1, p_num_patches,
                                                                                         self.original_embedding_dim).transpose(
                1, 2).reshape(1, self.original_embedding_dim, p_f_dim, p_t_dim)
            # cut or interpolate the positional embedding
            if t_dim < p_t_dim:
                new_pos_embed = new_pos_embed[:, :, :,
                                int(p_t_dim / 2) - int(t_dim / 2): int(p_t_dim / 2) - int(t_dim / 2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(8, t_dim), mode='bilinear')
            if f_dim < p_f_dim:
                new_pos_embed = new_pos_embed[:, :,
                                int(p_f_dim / 2) - int(f_dim / 2): int(p_f_dim / 2) - int(f_dim / 2) + t_dim, :]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

            new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(
                torch.cat([self.v.pos_embed[:, :self.cls_token_num, :].detach(), new_pos_embed], dim=1))

    # get the shape of intermediate representation.
    def get_shape(self, fstride, tstride, input_fdim, input_tdim, fshape, tshape):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(fshape, tshape), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    # generate mask for 16*16 patch
    def gen_maskid_patch(self, sequence_len=512, mask_size=100, cluster=3):
        mask_id = []

        # randomize clutering factor in [3,6)
        cur_clus = randrange(cluster) + 3

        while len(list(set(mask_id))) <= mask_size:
            start_id = randrange(sequence_len)

            # this improves the efficiency, but might change the pretrained model
            # while start_id in mask_id:
            #     start_id = randrange(sequence_len)

            cur_mask = []
            for i in range(0, cur_clus):
                for j in range(0, cur_clus):
                    mask_cand = start_id + self.p_t_dim * i + j
                    if mask_cand > 0 and mask_cand < sequence_len:
                        cur_mask.append(mask_cand)
            mask_id = mask_id + cur_mask
        mask_id = list(set(mask_id))[:mask_size]
        return torch.tensor(mask_id)

    # using cluster for frame masking hurts the performance, so just use the naive random sampling
    def gen_maskid_frame(self, sequence_len=512, mask_size=100):
        mask_id = random.sample(range(0, sequence_len), mask_size)
        return torch.tensor(mask_id)

    def finetuningavgtok(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)

        # average output of all tokens except cls token(s)
        x = torch.mean(x[:, self.cls_token_num:, :], dim=1)
        x = self.mlp_head(x)
        return x

    def finetuningcls(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        if self.cls_token_num == 2:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            dist_token = self.v.dist_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)
        else:
            cls_tokens = self.v.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk_id, blk in enumerate(self.v.blocks):
            x = blk(x)
        x = self.v.norm(x)

        # if models has two cls tokens (DEIT), average as the clip-level representation
        if self.cls_token_num == 2:
            x = (x[:, 0] + x[:, 1]) / 2
        else:
            x = x[:, 0]
        x = self.mlp_head(x)
        return x

    # masked patch pretraining with discriminative objective
    def mpc(self, x, mask_patch, cluster, show_mask=False):
        input = self.unfold(x).transpose(1, 2)
        B = x.shape[0]
        # x in shape (batch_size, sequence_len, embedding dim)
        x = self.v.patch_embed(x)

        # encode the patch
        # size 12(batch_size) * 100(#mask_patch) * 768(hidden_dim), prepare to save the true values of masked samples
        encode_samples = torch.empty((B, mask_patch, 256), device=x.device, requires_grad=False).float()
        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)

        # for each audio clip in the batch
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            # copy the masked embeddings, note gradients are stopped in this path
            encode_samples[i] = input[i, mask_index[i], :].clone().detach()
            # mask the encode samples with 0
            mask_dense[i, mask_index[i], :] = 0

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # mask the patch
        x = x * mask_dense + (1 - mask_dense) * mask_tokens

        # pass through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        # prediction of the masked patch
        pred = torch.empty((B, mask_patch, 256), device=x.device).float()  # e.g. size 12*100*768
        for i in range(B):
            #  +2 for indexes because skipping the cls and dis token
            # we map the output of transformer (768-dim for base models) to 256-dim patch input space, and then dot product with flattened patch input (also 256-dim) to calculate loss.
            # alternatively, you can map the output of transformer to 768-dim patch embedding space, and dot product with patch embedding. Performance-wise they are similar, but map to 256 space is more efficient.
            pred[i] = self.cpredlayer(x[i, mask_index[i] + self.cls_token_num, :])

        # calculate the NCE loss
        nce = torch.tensor(0.0).to(x.device)
        correct = torch.tensor(0.0).to(x.device)
        for i in np.arange(0, B):
            # negative samples are from the same batch
            # 8/12/2022: has a difference with equation (1) in the ssast paper but (likely) performance-wise similar, see https://github.com/YuanGongND/ssast/issues/13
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # e.g. size 100*100
            correct += torch.sum(torch.eq(torch.argmax(self.softmax(total), dim=0),
                                          torch.arange(0, mask_patch, device=x.device)))  # correct is a tensor
            nce += torch.sum(torch.diag(self.lsoftmax(total)))  # nce is a tensor
        acc = 1. * correct / (B * mask_patch)
        nce = nce / (-1. * B * mask_patch)

        # visualize the masked area, for probing test only, set show_mask = False for any training/inference.
        if show_mask == False:
            return acc, nce
        else:
            if B > 1:
                raise Exception('Currently only support single spectrogram probing test.')

            self.mask_correct = torch.nn.Parameter(torch.arange(0, mask_patch), requires_grad=False)

            pred = input.clone()  # [B, 512, 256]
            masked = input.clone()

            for i in range(B):
                result = [float(t) * 99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)]
                pred[i, mask_index[i], :] = torch.tensor(result).reshape(mask_patch, 1).expand(mask_patch, 256)
                masked[i, mask_index[i], :] = 99.0

            # print(total)
            # print(self.softmax(total))
            # print(torch.argmax(self.softmax(total), dim=0))
            # print(self.mask_correct)
            # print(torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct))
            # print([float(t)*99 for t in torch.eq(torch.argmax(self.softmax(total), dim=0), self.mask_correct)])

            fold = torch.nn.Fold(output_size=([self.input_fdim, self.input_tdim]),
                                 kernel_size=(self.fshape, self.tshape), stride=(self.fstride, self.tstride))
            pred = fold(pred.transpose(1, 2))
            masked = fold(masked.transpose(1, 2))

            return pred, masked

    # # masked patch pretraining with generative objective
    def mpg(self, input, mask_patch, cluster):
        B = input.shape[0]
        x = self.v.patch_embed(input)
        input = self.unfold(input).transpose(1, 2)

        # size 12(batch_size) * 100(#mask_patch), index of masked patches
        mask_index = torch.empty((B, mask_patch), device=x.device, requires_grad=False).long()
        # size 12(batch_size) * 512(sequence_len) * 768(hidden_dim)
        mask_dense = torch.ones([x.shape[0], x.shape[1], x.shape[2]], device=x.device)
        for i in range(B):
            # randomly generate #mask_patch mask indexes without duplicate
            if cluster == True:
                # use this if you are masking e.g. 16*16 patches
                mask_index[i] = self.gen_maskid_patch(self.num_patches, mask_patch)
            else:
                # use this if you are masking frame, i.e., 128*2 patches
                mask_index[i] = self.gen_maskid_frame(self.num_patches, mask_patch)
            mask_dense[i, mask_index[i], :] = 0

        mask_tokens = self.mask_embed.expand(B, x.shape[1], -1)

        # follow BEIT paper, mask with learnable masking embedding, but no performance diff observed compared with masking with 0s.
        x = x * mask_dense + (1 - mask_dense) * mask_tokens

        # go through the Transformer layers
        cls_tokens = self.v.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)
        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x)

        pred = torch.empty((B, mask_patch, self.fshape * self.tshape), device=x.device).float()  # e.g. size 12*100*256
        target = torch.empty((B, mask_patch, self.fshape * self.tshape),
                             device=x.device).float()  # e.g. size 12*100*256

        for i in range(B):
            #  +2 for indexes because cls and dis token
            pred[i] = self.gpredlayer(x[i, mask_index[i] + self.cls_token_num, :])
            target[i] = input[i, mask_index[i], :]

        # calculate the MSE loss
        mse = torch.mean((pred - target) ** 2)

        return mse

    def forward(self, x, task, cluster=True, mask_patch=400):
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        # finetuning (ft), use the mean of all token (patch) output as clip-level representation.
        # this is default for SSAST fine-tuning as during pretraining, supervision signal is given to each token, not the [cls] token
        if task == 'ft_avgtok':
            return self.finetuningavgtok(x)
        # alternatively, use the [cls] token output as clip-level representation.
        elif task == 'ft_cls':
            return self.finetuningcls(x)
        # pretraining, masked patch classification (discriminative objective)
        elif task == 'pretrain_mpc':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster)
        # pretraining, masked patch reconstruction (generative objective)
        elif task == 'pretrain_mpg':
            return self.mpg(x, mask_patch=mask_patch, cluster=cluster)
        elif task == 'visualize_mask':
            return self.mpc(x, mask_patch=mask_patch, cluster=cluster, show_mask=True)
        else:
            raise Exception('Task unrecognized.')


########################################################################################################################
""" 
    PaSST (The Patchout faSt Spectrogram Transformer (PaSST)) 
    https://github.com/kkoutini/PaSST && https://github.com/kkoutini/passt_hear21 
    
    Apache License 2.0
"""


# PaSST:

class PasstBasicWrapper(nn.Module, audobject.Object):
    """
    @param mel: spectrogram extractor
    @param net: network module
    @param max_model_window: maximum clip length allowed by the model (milliseconds).
    @param timestamp_hop: the hop lengh for timestamp embeddings (milliseconds).
    @param scene_hop: the hop lengh for scene embeddings (milliseconds).
    @param scene_embedding_size:
    @param timestamp_embedding_size:
    @param mode: "all", "embed_only", "logits"
    """

    @audobject.init_decorator(hide=['mel', 'net'])
    def __init__(
            self,
            mel=None,
            net=None,
            max_model_window=10000,
            timestamp_window=160,
            timestamp_hop=50,
            scene_hop=2500,
            scene_embedding_size=1295,
            timestamp_embedding_size=1295,
            mode="all",
            prespec=False,  # decices whether or not to receive waveforms or melspect directly, we can use preprocessed
    ):

        torch.nn.Module.__init__(self)
        assert timm.__version__ == '0.4.12', 'Please use timm == 0.4.12, the code might not be compatible with newer versions.'
        self.device_proxy = nn.Parameter(torch.zeros((1)))
        self.sample_rate = mel.sr
        self.timestamp_window = int(timestamp_window * self.sample_rate / 1000)
        self.max_model_window = int(max_model_window * self.sample_rate / 1000)
        self.timestamp_hop = int(timestamp_hop * self.sample_rate / 1000)
        self.scene_hop = int(scene_hop * self.sample_rate / 1000)
        self.scene_embedding_size = scene_embedding_size
        self.timestamp_embedding_size = timestamp_embedding_size
        self.mode = mode
        self.mel = mel
        self.net = net
        self.prespec = prespec

    def device(self):
        return self.device_proxy.device

    def forward(self, x):

        if not self.prespec:  # for preprocessed spectrograms
            x = self.mel(x)

        if x.ndim == 3:  # waveform input
            x = x.unsqueeze(1)

        if self.prespec:
            x = x.permute(0, 1, 3, 2)

        x, features = self.net(x)
        if self.mode == "all":
            embed = torch.cat([x, features], dim=1)
        elif self.mode == "embed_only":
            embed = features
        elif self.mode == "logits":
            embed = x
        else:
            raise RuntimeError(f"mode='{self.mode}' is not recognized not in: all, embed_only, logits")
        return embed

    def get_scene_embeddings(self, audio):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        model: Loaded Model.
        Returns:
        embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
        """
        n_sounds, n_samples = audio.shape
        if n_samples <= self.max_model_window:
            embed = self.forward(audio.to(self.device()).contiguous())
            return embed
        embeddings, timestamps = self.get_timestamp_embeddings(audio, window_size=self.max_model_window,
                                                               hop=self.scene_hop)
        return embeddings.mean(axis=1)

    def get_timestamp_embeddings(self, audio: torch.Tensor, window_size=None, hop=None, pad=None):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        model: Loaded Model.
        Returns:
        embedding: A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
        timestamps: A float32 Tensor with shape (`n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
        """
        if hop is None:
            hop = self.timestamp_hop
        if window_size is None:
            window_size = self.timestamp_window
        if pad is None:
            pad = window_size // 2
        audio = audio.cpu()
        n_sounds, n_samples = audio.shape
        audio = audio.unsqueeze(1)  # n_sounds,1, (n_samples+pad*2)
        # print(audio.shape)
        padded = F.pad(audio, (pad, pad), mode='reflect')
        # print(padded.shape)
        padded = padded.unsqueeze(1)  # n_sounds,1, (n_samples+pad*2)
        # print(padded.shape)
        segments = F.unfold(padded, kernel_size=(1, window_size), stride=(1, hop)).transpose(-1, -2).transpose(0, 1)
        timestamps = []
        embeddings = []
        for i, segment in enumerate(segments):
            timestamps.append(i)
            embeddings.append(self.forward(segment.to(self.device())).cpu())
        timestamps = torch.as_tensor(timestamps) * hop * 1000. / self.sample_rate

        embeddings = torch.stack(embeddings).transpose(0, 1)  # now n_sounds, n_timestamps, timestamp_embedding_size
        timestamps = timestamps.unsqueeze(0).expand(n_sounds, -1)

        return embeddings, timestamps

    def get_timestamp_mels(self, audio: torch.Tensor, window_size=None, hop=None, pad=None):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        model: Loaded Model.
        Returns:
        embedding: A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
        timestamps: A float32 Tensor with shape (`n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
        """
        if hop is None:
            hop = self.timestamp_hop
        if window_size is None:
            window_size = self.timestamp_window
        if pad is None:
            pad = window_size // 2
        audio = audio.cpu()
        n_sounds, n_samples = audio.shape
        audio = audio.unsqueeze(1)  # n_sounds,1, (n_samples+pad*2)
        # print(audio.shape)
        padded = F.pad(audio, (pad, pad), mode='reflect')
        # print(padded.shape)
        padded = padded.unsqueeze(1)  # n_sounds,1, (n_samples+pad*2)
        # print(padded.shape)
        segments = F.unfold(padded, kernel_size=(1, window_size), stride=(1, hop)).transpose(-1, -2).transpose(0, 1)
        timestamps = []
        embeddings = []
        for i, segment in enumerate(segments):
            timestamps.append(i)
            embeddings.append(self.mel(segment.to(self.device())).cpu().reshape(n_sounds, 128 * 6))
        timestamps = torch.as_tensor(timestamps) * hop * 1000. / self.sample_rate

        embeddings = torch.stack(embeddings).transpose(0, 1)  # now n_sounds, n_timestamps, timestamp_embedding_size
        timestamps = timestamps.unsqueeze(0).expand(n_sounds, -1)

        return embeddings, timestamps


########################################################################################################################
"""
    HTS (Hierarchical Token Semantic Audio Transformer)
    https://github.com/RetroCirce/HTS-Audio-Transformer
    
    MIT license
"""


# HTS-AT:

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple_hts = _ntuple(2)


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_hts(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_hts(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbedHTS(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True,
                 patch_stride=16):
        super().__init__()
        img_size = to_2tuple_hts(img_size)
        patch_size = to_2tuple_hts(patch_size)
        patch_stride = to_2tuple_hts(patch_stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        padding = ((patch_size[0] - patch_stride[0]) // 2, (patch_size[1] - patch_stride[1]) // 2)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # print(x.shape)  # torch.Size([32, 1, 256, 256])
        # print(self.img_size)  # (256, 256)
        x = self.proj(x)
        # print(x.shape)  # torch.Size([32, 96, 64, 64])
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        # print(x.shape)  # torch.Size([32, 4096, 96])
        # exit(0)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_hts(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# We use the model based on Swintransformer Block, therefore we can use the swin-transformer pretrained model
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_before_mlp='ln'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm_before_mlp = norm_before_mlp
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple_hts(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.norm_before_mlp == 'ln':
            self.norm2 = nn.LayerNorm(dim)
        elif self.norm_before_mlp == 'bn':
            self.norm2 = lambda x: nn.BatchNorm1d(dim)(x.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # pdb.set_trace()
        H, W = self.input_resolution
        # print("H: ", H)
        # print("W: ", W)
        # pdb.set_trace()
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows, attn = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 norm_before_mlp='ln'):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, norm_before_mlp=norm_before_mlp)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        attns = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, attn = blk(x)
                if not self.training:
                    attns.append(attn.unsqueeze(0))
        if self.downsample is not None:
            x = self.downsample(x)
        if not self.training:
            attn = torch.cat(attns, dim=0)
            attn = torch.mean(attn, dim=0)
        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


# The Core of HTSAT
class HTSAT_Swin_Transformer(nn.Module):
    r"""HTSAT based on the Swin Transformer
    Args:
        spec_size (int | tuple(int)): Input Spectrogram size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 4
        path_stride (iot | tuple(int)): Patch Stride for Frequency and Time Axis. Default: 4
        in_chans (int): Number of input image channels. Default: 1 (mono)
        num_classes (int): Number of classes for classification head. Default: 527
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each HTSAT-Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        config (module): The configuration Module from config.py
    """

    def __init__(
            self,
            spec_size=256,
            patch_size=4,
            patch_stride=(4, 4),
            in_chans=1,
            num_classes=1,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[4, 8, 16, 32],
            window_size=5,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            norm_before_mlp='ln',
            config=None,
            sigmoid_output=False,
            **kwargs
    ):
        super(HTSAT_Swin_Transformer, self).__init__()
        self.config = config
        self.spec_size = spec_size
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.ape = ape
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))

        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale

        self.patch_norm = patch_norm
        self.norm_layer = norm_layer if self.patch_norm else None
        self.norm_before_mlp = norm_before_mlp
        self.sigmoid_output = sigmoid_output
        self.mlp_ratio = mlp_ratio

        self.use_checkpoint = use_checkpoint

        #  process mel-spec ; used only once
        self.freq_ratio = self.spec_size // self.config.mel_bins
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32  # Downsampled ratio

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=config.window_size, hop_length=config.hop_size,
                                                 win_length=config.window_size, window=window, center=center,
                                                 pad_mode=pad_mode,
                                                 freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=config.sample_rate, n_fft=config.window_size,
                                                 n_mels=config.mel_bins, fmin=config.fmin, fmax=config.fmax, ref=ref,
                                                 amin=amin, top_db=top_db,
                                                 freeze_parameters=True)
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)  # 2 2

        self.filter_augmenter = FilterAugment()

        self.bn0 = nn.BatchNorm2d(self.config.mel_bins)

        # split spctrogram into non-overlapping patches
        self.patch_embed = PatchEmbedHTS(
            img_size=self.spec_size, patch_size=self.patch_size, in_chans=self.in_chans,
            embed_dim=self.embed_dim, norm_layer=self.norm_layer, patch_stride=patch_stride)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_hts(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # stochastic depth
        dpr = [x.item() for x in
               torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=self.depths[i_layer],
                               num_heads=self.num_heads[i_layer],
                               window_size=self.window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                               drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                               drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                               norm_layer=self.norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               norm_before_mlp=self.norm_before_mlp)
            self.layers.append(layer)

        # A deprecated optimization for using a hierarchical output from different blocks
        # if self.config.htsat_hier_output:
        #     self.norm = nn.ModuleList(
        #         [self.norm_layer(
        #             min(
        #               self.embed_dim * (2 ** (len(self.depths) - 1)),
        #               self.embed_dim * (2 ** (i + 1))
        #                 )
        #         ) for i in range(len(self.depths))]
        #     )
        # else:

        self.norm = self.norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        # A deprecated optimization for using the max value instead of average value
        # if self.config.htsat_use_max:
        #     self.a_avgpool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        #     self.a_maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        if self.config.enable_tscam:
            # if self.config.htsat_hier_output:
            #     self.tscam_conv = nn.ModuleList()
            #     for i in range(len(self.depths)):
            #         zoom_ratio = 2 ** min(len(self.depths) - 1, i + 1)
            #         zoom_dim = min(
            #             self.embed_dim * (2 ** (len(self.depths) - 1)),
            #             self.embed_dim * (2 ** (i + 1))
            #         )
            #         SF = self.spec_size // zoom_ratio // self.patch_stride[0] // self.freq_ratio
            #         self.tscam_conv.append(
            #             nn.Conv2d(
            #                 in_channels = zoom_dim,
            #                 out_channels = self.num_classes,
            #                 kernel_size = (SF, 3),
            #                 padding = (0,1)
            #             )
            #         )
            #     self.head = nn.Linear(num_classes * len(self.depths), num_classes)
            # else:

            SF = self.spec_size // (2 ** (len(self.depths) - 1)) // self.patch_stride[0] // self.freq_ratio
            self.tscam_conv = nn.Conv2d(
                in_channels=self.num_features,
                out_channels=self.num_classes,
                kernel_size=(SF, 3),
                padding=(0, 1)
            )
            self.head = nn.Linear(num_classes, num_classes)
        else:
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_hts(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        # A deprecated optimization for using a hierarchical output from different blocks
        # if self.config.htsat_hier_output:
        #     hier_x = []
        #     hier_attn = []

        frames_num = x.shape[2]
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)
            # A deprecated optimization for using a hierarchical output from different blocks
            # if self.config.htsat_hier_output:
            #     hier_x.append(x)
            #     if i == len(self.layers) - 1:
            #         hier_attn.append(attn)

        # A deprecated optimization for using a hierarchical output from different blocks
        # if self.config.htsat_hier_output:
        #     hxs = []
        #     fphxs = []
        #     for i in range(len(hier_x)):
        #         hx = hier_x[i]
        #         hx = self.norm[i](hx)
        #         B, N, C = hx.shape
        #         zoom_ratio = 2 ** min(len(self.depths) - 1, i + 1)
        #         SF = frames_num // zoom_ratio // self.patch_stride[0]
        #         ST = frames_num // zoom_ratio // self.patch_stride[1]
        #         hx = hx.permute(0,2,1).contiguous().reshape(B, C, SF, ST)
        #         B, C, F, T = hx.shape
        #         c_freq_bin = F // self.freq_ratio
        #         hx = hx.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
        #         hx = hx.permute(0,1,3,2,4).contiguous().reshape(B, C, c_freq_bin, -1)

        #         hx = self.tscam_conv[i](hx)
        #         hx = torch.flatten(hx, 2)
        #         fphx = interpolate(hx.permute(0,2,1).contiguous(), self.spec_size * self.freq_ratio // hx.shape[2])

        #         hx = self.avgpool(hx)
        #         hx = torch.flatten(hx, 1)
        #         hxs.append(hx)
        #         fphxs.append(fphx)
        #     hxs = torch.cat(hxs, dim=1)
        #     fphxs = torch.cat(fphxs, dim = 2)
        #     hxs = self.head(hxs)
        #     fphxs = self.head(fphxs)
        #     output_dict = {'framewise_output': torch.sigmoid(fphxs),
        #         'clipwise_output': torch.sigmoid(hxs)}
        #     return output_dict

        if self.config.enable_tscam:
            # for x
            x = self.norm(x)
            B, N, C = x.shape
            SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
            ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
            x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
            B, C, F, T = x.shape
            # group 2D CNN
            c_freq_bin = F // self.freq_ratio
            x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)

            # get latent_output
            latent_output = self.avgpool(torch.flatten(x, 2))
            latent_output = torch.flatten(latent_output, 1)

            # display the attention map, if needed
            if self.config.htsat_attn_heatmap:
                # for attn
                attn = torch.mean(attn, dim=1)
                attn = torch.mean(attn, dim=1)
                attn = attn.reshape(B, SF, ST)
                c_freq_bin = SF // self.freq_ratio
                attn = attn.reshape(B, SF // c_freq_bin, c_freq_bin, ST)
                attn = attn.permute(0, 2, 1, 3).contiguous().reshape(B, c_freq_bin, -1)
                attn = attn.mean(dim=1)
                attn_max = torch.max(attn, dim=1, keepdim=True)[0]
                attn_min = torch.min(attn, dim=1, keepdim=True)[0]
                attn = ((attn * 0.15) + (attn_max * 0.85 - attn_min)) / (attn_max - attn_min)
                attn = attn.unsqueeze(dim=2)

            x = self.tscam_conv(x)
            x = torch.flatten(x, 2)  # B, C, T

            # A deprecated optimization for using the max value instead of average value
            # if self.config.htsat_use_max:
            #     x1 = self.a_maxpool(x)
            #     x2 = self.a_avgpool(x)
            #     x = x1 + x2

            if self.config.htsat_attn_heatmap:
                x = torch.sigmoid(x) if self.sigmoid_output else x
                fpx = interpolate(x.permute(0, 2, 1).contiguous() * attn, 8 * self.patch_stride[1])
            else:
                x = torch.sigmoid(x) if self.sigmoid_output else x
                fpx = interpolate(x.permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1])

                # A deprecated optimization for using the max value instead of average value
            # if self.config.htsat_use_max:
            #     x1 = self.avgpool(x)
            #     x2 = self.maxpool(x)
            #     x = x1 + x2
            # else:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)

            if self.config.loss_type == "clip_ce":
                output_dict = {
                    'framewise_output': fpx,  # already sigmoided
                    'clipwise_output': x,
                    'latent_output': latent_output
                }
            else:
                output_dict = {
                    'framewise_output': fpx,  # already sigmoided
                    'clipwise_output': torch.sigmoid(x) if self.sigmoid_output else x,
                    'latent_output': latent_output
                }

        else:
            x = self.norm(x)  # B N C
            B, N, C = x.shape
            fpx = x.permute(0, 2, 1).contiguous().reshape(B, C, frames_num // (2 ** (len(self.depths) + 1)),
                                                          frames_num // (2 ** (len(self.depths) + 1)))
            B, C, F, T = fpx.shape
            c_freq_bin = F // self.freq_ratio
            fpx = fpx.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
            fpx = fpx.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
            fpx = torch.sum(fpx, dim=2)
            fpx = interpolate(fpx.permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1])
            x = self.avgpool(x.transpose(1, 2))  # B C 1
            x = torch.flatten(x, 1)
            if self.num_classes > 0:
                x = self.head(x)
                fpx = self.head(fpx)
            output_dict = {'framewise_output': torch.sigmoid(fpx) if self.sigmoid_output else fpx,
                           'clipwise_output': torch.sigmoid(x) if self.sigmoid_output else x}
        return output_dict

    def crop_wav(self, x, crop_size, spe_pos=None):
        time_steps = x.shape[2]
        tx = torch.zeros(x.shape[0], x.shape[1], crop_size, x.shape[3]).to(x.device)
        for i in range(len(x)):
            if spe_pos is None:
                crop_pos = random.randint(0, time_steps - crop_size - 1)
            else:
                crop_pos = spe_pos
            tx[i][0] = x[i, 0, crop_pos:crop_pos + crop_size, :]
        return tx

    # Reshape the wavform to a img size, if you want to use the pretrained swin transformer model
    def reshape_wav2img(self, x):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and F <= target_F, "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.freq_ratio, x.shape[3] // self.freq_ratio)
        # print(x.shape)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
        return x

    # Repeat the wavform to a img size, if you want to use the pretrained swin transformer model
    def repeat_wat2img(self, x, cur_pos):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and F <= target_F, "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)
        x = x.permute(0, 1, 3, 2).contiguous()  # B C F T
        x = x[:, :, :, cur_pos:cur_pos + self.spec_size]
        x = x.repeat(repeats=(1, 1, 4, 1))
        return x

    def forward(self, x: torch.Tensor, mixup_lambda=None, infer_mode=False):  # out_feat_keys: List[str] = None):
        """ in : (B, SR * WDW[s]), i.e. (32, 80000)"""

        # x = self.spectrogram_extractor(x)  # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)

        if x.ndim == 2:  # waveform input, ie [32, 80000]
            transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=320,  # 320
                f_min=50,  # 50
                f_max=8000,  # 8000
                n_mels=128)  # 64
            x = transform(x.cpu())  # [32, 64, 49x]
            x = x.permute(0, 2, 1)  # [32, 49x, 64]
            x = x[:, None, :]  # add channel [32, 1, 49x, 64]
            transform = audtorch.transforms.RandomCrop(500, axis=-2)  # pad to 500 [32, 1, 500, 64]
            x = transform(x).cuda()

        x = x.transpose(1, 3)
        x = self.bn0(x)
        x = x.transpose(1, 3)

        if self.training:
            # x = self.spec_augmenter(x)
            x = self.filter_augmenter(x)
        if self.training and mixup_lambda is not None:
            x = do_mixup(x, mixup_lambda)

        if infer_mode:
            # in infer mode. we need to handle different length audio input
            frame_num = x.shape[2]
            target_T = int(self.spec_size * self.freq_ratio)
            repeat_ratio = math.floor(target_T / frame_num)
            x = x.repeat(repeats=(1, 1, repeat_ratio, 1))
            x = self.reshape_wav2img(x)
            output_dict = self.forward_features(x)
        elif self.config.enable_repeat_mode:
            if self.training:
                cur_pos = random.randint(0, (self.freq_ratio - 1) * self.spec_size - 1)
                x = self.repeat_wat2img(x, cur_pos)
                output_dict = self.forward_features(x)
            else:
                output_dicts = []
                for cur_pos in range(0, (self.freq_ratio - 1) * self.spec_size + 1, self.spec_size):
                    tx = x.clone()
                    tx = self.repeat_wat2img(tx, cur_pos)
                    output_dicts.append(self.forward_features(tx))
                clipwise_output = torch.zeros_like(output_dicts[0]["clipwise_output"]).float().to(x.device)
                framewise_output = torch.zeros_like(output_dicts[0]["framewise_output"]).float().to(x.device)
                for d in output_dicts:
                    clipwise_output += d["clipwise_output"]
                    framewise_output += d["framewise_output"]
                clipwise_output = clipwise_output / len(output_dicts)
                framewise_output = framewise_output / len(output_dicts)

                output_dict = {
                    'framewise_output': framewise_output,
                    'clipwise_output': clipwise_output
                }
        else:
            if x.shape[2] > self.freq_ratio * self.spec_size:
                if self.training:
                    x = self.crop_wav(x, crop_size=self.freq_ratio * self.spec_size)
                    x = self.reshape_wav2img(x)
                    output_dict = self.forward_features(x)
                else:
                    # Change: Hard code here
                    overlap_size = (x.shape[2] - 1) // 4
                    output_dicts = []
                    crop_size = (x.shape[2] - 1) // 2
                    for cur_pos in range(0, x.shape[2] - crop_size - 1, overlap_size):
                        tx = self.crop_wav(x, crop_size=crop_size, spe_pos=cur_pos)
                        tx = self.reshape_wav2img(tx)
                        output_dicts.append(self.forward_features(tx))
                    clipwise_output = torch.zeros_like(output_dicts[0]["clipwise_output"]).float().to(x.device)
                    framewise_output = torch.zeros_like(output_dicts[0]["framewise_output"]).float().to(x.device)
                    for d in output_dicts:
                        clipwise_output += d["clipwise_output"]
                        framewise_output += d["framewise_output"]
                    clipwise_output = clipwise_output / len(output_dicts)
                    framewise_output = framewise_output / len(output_dicts)
                    output_dict = {
                        'framewise_output': framewise_output,
                        'clipwise_output': clipwise_output
                    }
            else:  # this part is typically used, and most easy one
                x = self.reshape_wav2img(
                    x)  # converts spectrogram image (32, 1, 64, 500) to fit the ViT (32, 1, 256, 256)
                output_dict = self.forward_features(x)
        # x = self.head(x)
        # return output_dict
        return output_dict['clipwise_output']


########################################################################################################################


if __name__ == '__main__':

    ast = False
    sasst = False
    passt = True
    hts = False

    if ast:
        b_size = 16
        input_tdim = 500
        input_fdim = 64
        num_classes = 1

        ast_mdl = ASTModel(input_tdim=input_tdim, input_fdim=input_fdim, label_dim=num_classes, verbose=True)
        test_input = torch.rand([b_size, input_tdim, input_fdim])
        test_output = ast_mdl(test_input)  # input a batch

        print('Output-shape:', test_output.shape)
        print('Output-value:', test_output)

    if sasst:
        # Reference: https://github.com/YuanGongND/ssast/blob/main/src/models/ast_models.py
        sasst_unlabeled = False
        input_tdim = 500  # == windows_size
        input_fdim = 64  # == mel_bins
        batch_size = 32
        label_dim = 1  # == num_classes

        # label_dim = 527,
        # fshape = 128,
        # tshape = 2,
        # fstride = 128,
        # tstride = 2,
        # input_fdim = 128,
        # input_tdim = 1024,
        # model_size = 'base',
        # pretrain_stage = True,
        # load_pretrained_mdl_path = None

        if sasst_unlabeled:
            # pretraining stage
            # create a 16*16 patch based SSAST model for pretraining.
            # note, we don't use patch split overlap in pretraining, so fstride=fshape and tstride=tshape
            ast_mdl = SSASTModel(fshape=16, tshape=16, fstride=16, tstride=16,
                                 input_fdim=input_fdim, input_tdim=input_tdim, model_size='base',
                                 pretrain_stage=True)

            # # alternatively, create a frame based AST model
            # ast_mdl = SSASTModel(
            #              fshape=128, tshape=2, fstride=128, tstride=2,
            #              input_fdim=128, input_tdim=input_tdim, model_size='base',
            #              pretrain=True)

            # do pretraining, see src/traintest_mask.py for our full pretraining code
            # input in shape [batch_size, input_tdim, input_fdim]
            test_input = torch.zeros([10, input_tdim, 128])
            # mask 100 patches for both discriminative and generative loss
            acc, nce_loss = ast_mdl(test_input, task='pretrain_mpc', mask_patch=100)
            mse_loss = ast_mdl(test_input, task='pretrain_mpg', mask_patch=100)
            loss = nce_loss + 10 * mse_loss
            # do back propagate and update the model, etc

            # after pretraining, save the pretrained model.
            # the code is designed for Dataparallel model
            ast_mdl = torch.nn.DataParallel(ast_mdl)
            torch.save(ast_mdl.state_dict(), './test_mdl.pth')
        else:
            # fine-tuning stage
            # now you have a labeled dataset you want to finetune AST on
            # suppose the avg length is 100 frames (1s) and there are 35 classes
            # the fshape and tshape must be same in pretraining and finetuning
            # but fstride and tstride can be different in pretraining and finetuning
            # using smaller strides improves the performance but also increase the computational overhead
            # set pretrain_stage as False since now is in the finetuning stage
            # provide the path of the pretrained model you want to load
            # input_tdim = 100  # fine-tuning data length can be different with pretraining data length

            ckp_path = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\models\checkpoints\sasst\SSAST-Base-Patch-400.pth'

            ast_mdl = SSASTModel(label_dim=label_dim, fshape=16, tshape=16, fstride=10, tstride=10,
                                 input_fdim=input_fdim, input_tdim=input_tdim, model_size='base',
                                 pretrain_stage=False, load_pretrained_mdl_path=ckp_path)

            # # alternatively, use a frame based AST model
            # ast_mdl = SSASTModel(label_dim=label_dim,
            #                      fshape=16, tshape=16, fstride=128, tstride=1,
            #                      input_fdim=input_fdim, input_tdim=input_tdim, model_size='base',
            #                      pretrain_stage=False, load_pretrained_mdl_path=ckp_path)

            # do finetuning, see src/traintest.py for our finetuning code
            test_input = torch.zeros([batch_size, input_tdim, input_fdim])  # [16, 500, 64]
            prediction = ast_mdl(test_input, task='ft_avgtok')
            print(prediction.shape)

            # calculate the loss, do back propagate, etc
            # # (optional) do some probe test
            # test_input = torch.zeros([1, input_tdim, 128]).to(device)
            # acc, nce = ast_mdl(test_input, task='pretrain_mpc', mask_patch=100)
            # # you can visualize the mask
            # pred, masked = ast_mdl(test_input, task='visualize_mask', mask_patch=100)
            # plt.imshow(masked[0,0])
            # plt.show()

    if passt:
        b_size = 16
        sr = 16000

        wvpath = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\waveform_data\wdw5\fold0\000000000002.npy'
        waveform = np.load(wvpath)  # (1, 80144), sr*sec=80000
        waveform = torch.Tensor(waveform).cuda()  # torch.Size([1, 80144])

        # get the PaSST model wrapper, includes Melspectrogram and the default pre-trained transformer
        model = get_basic_model(mode="logits")
        # print(model.mel)  # Extracts mel spectrogram from raw waveforms.
        # print(model.net)  # the transformer network.

        # example inference
        model.eval()
        model = model.cuda()
        with torch.no_grad():
            # audio_wave has the shape of [batch, seconds*32000] sampling rate is 32k
            # example audio_wave of batch=3 and 10 seconds
            # audio = torch.ones((16, 32000 * 10)) * 0.5
            # audio_wave = audio.cuda()
            logits = model(waveform)
            from torchsummary import summary

            summary(model.cuda(), input_size=waveform.shape)
            print(logits.shape)
            print(logits)

    if hts:
        wvpath = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\waveform_data\wdw5\fold0\000000000002.npy'
        wvpath2 = r'C:\Software\Python\Bachelorarbeit\kirun-repo\kirun-speed-estimation\data\waveform_data\wdw5\fold0\000000000002.npy'
        waveform1 = np.load(wvpath)  # (1, 80144), sr*sec=80000
        waveform1 = torch.Tensor(waveform1).cuda()  # torch.Size([1, 80144])
        waveform2 = np.load(wvpath)  # (1, 80144), sr*sec=80000
        waveform2 = torch.Tensor(waveform2).cuda()  # torch.Size([1, 80144])

        waveform_batch = torch.cat([waveform1, waveform2], dim=0)

        sed_model = HTSAT_Swin_Transformer(
            spec_size=config.htsat_spec_size,
            patch_size=config.htsat_patch_size,
            in_chans=1,
            num_classes=config.classes_num,
            window_size=config.htsat_window_size,
            config=config,
            depths=config.htsat_depth,
            embed_dim=config.htsat_dim,
            patch_stride=config.htsat_stride,
            num_heads=config.htsat_num_head
        )

        # example inference
        sed_model.eval()
        sed_model = sed_model.cuda()
        with torch.no_grad():
            outdict = sed_model(waveform_batch)
            print('framewise_output', outdict['framewise_output'].shape)
            print('clipwise_output', outdict['clipwise_output'].shape)
