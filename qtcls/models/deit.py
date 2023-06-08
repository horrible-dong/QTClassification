# -------------------------------------------------------------------------------
# Modified from timm
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# -------------------------------------------------------------------------------

import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from .vision_transformer_timm import VisionTransformer

__all__ = [
    'VisionTransformerDistilled',
    'deit_tiny_patch16_224',
    'deit_small_patch16_224',
    'deit_base_patch16_224',
    'deit_base_patch16_384',
    'deit_tiny_distilled_patch16_224',
    'deit_small_distilled_patch16_224',
    'deit_base_distilled_patch16_224',
    'deit_base_distilled_patch16_384',
    'deit3_small_patch16_224',
    'deit3_small_patch16_384',
    'deit3_medium_patch16_224',
    'deit3_base_patch16_224',
    'deit3_base_patch16_384',
    'deit3_large_patch16_224',
    'deit3_large_patch16_384',
    'deit3_huge_patch14_224',
    'deit3_small_patch16_224_in21ft1k',
    'deit3_small_patch16_384_in21ft1k',
    'deit3_medium_patch16_224_in21ft1k',
    'deit3_base_patch16_224_in21ft1k',
    'deit3_base_patch16_384_in21ft1k',
    'deit3_large_patch16_224_in21ft1k',
    'deit3_large_patch16_384_in21ft1k',
    'deit3_huge_patch14_224_in21ft1k'
]


class VisionTransformerDistilled(VisionTransformer):
    """ Vision Transformer w/ Distillation Token and Head

    Distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, *args, **kwargs):
        weight_init = kwargs.pop('weight_init', '')
        super().__init__(*args, **kwargs, weight_init='skip')
        assert self.global_pool in ('token',)

        self.num_prefix_tokens = 2
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + self.num_prefix_tokens, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()
        self.distilled_training = False  # must set this True to train w/ distillation token

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        trunc_normal_(self.dist_token, std=.02)
        super().init_weights(mode=mode)

    def forward_features(self, x) -> torch.Tensor:
        x = self.patch_embed(x)
        x = torch.cat((
            self.cls_token.expand(x.shape[0], -1, -1),
            self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False) -> torch.Tensor:
        if pre_logits:
            return (x[:, 0] + x[:, 1]) / 2
        x, x_dist = self.head(x[:, 0]), self.head_dist(x[:, 1])
        if self.distilled_training and self.training and not torch.jit.is_scripting():
            # only return separate classification predictions when training in distilled mode
            return x, x_dist
        else:
            # during standard train / finetune, inference average the classifier predictions
            return (x + x_dist) / 2


def deit_tiny_patch16_224(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=192,
                                       depth=12,
                                       num_heads=3,
                                       **kwargs)
    return model


def deit_small_patch16_224(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=384,
                                       depth=12,
                                       num_heads=6,
                                       **kwargs)
    return model


def deit_base_patch16_224(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=768,
                                       depth=12,
                                       num_heads=12,
                                       **kwargs)
    return model


def deit_base_patch16_384(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=768,
                                       depth=12,
                                       num_heads=12,
                                       **kwargs)
    return model


def deit_tiny_distilled_patch16_224(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=192,
                                       depth=12,
                                       num_heads=3,
                                       **kwargs)
    return model


def deit_small_distilled_patch16_224(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=384,
                                       depth=12,
                                       num_heads=6,
                                       **kwargs)
    return model


def deit_base_distilled_patch16_224(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=768,
                                       depth=12,
                                       num_heads=12,
                                       **kwargs)
    return model


def deit_base_distilled_patch16_384(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=768,
                                       depth=12,
                                       num_heads=12,
                                       **kwargs)
    return model


def deit3_small_patch16_224(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=384,
                                       depth=12,
                                       num_heads=6,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_small_patch16_384(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=384,
                                       depth=12,
                                       num_heads=6,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_medium_patch16_224(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=512,
                                       depth=12,
                                       num_heads=8,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_base_patch16_224(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=768,
                                       depth=12,
                                       num_heads=12,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_base_patch16_384(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=768,
                                       depth=12,
                                       num_heads=12,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_large_patch16_224(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=1024,
                                       depth=24,
                                       num_heads=16,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_large_patch16_384(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=1024,
                                       depth=24,
                                       num_heads=16,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_huge_patch14_224(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=14,
                                       embed_dim=1280,
                                       depth=32,
                                       num_heads=16,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_small_patch16_224_in21ft1k(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=384,
                                       depth=12,
                                       num_heads=6,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_small_patch16_384_in21ft1k(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=384,
                                       depth=12,
                                       num_heads=6,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_medium_patch16_224_in21ft1k(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=512,
                                       depth=12,
                                       num_heads=8,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_base_patch16_224_in21ft1k(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=768,
                                       depth=12,
                                       num_heads=12,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_base_patch16_384_in21ft1k(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=768,
                                       depth=12,
                                       num_heads=12,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_large_patch16_224_in21ft1k(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=1024,
                                       depth=24,
                                       num_heads=16,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_large_patch16_384_in21ft1k(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=16,
                                       embed_dim=1024,
                                       depth=24,
                                       num_heads=16,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model


def deit3_huge_patch14_224_in21ft1k(**kwargs):
    model = VisionTransformerDistilled(img_size=224,
                                       in_chans=3,
                                       patch_size=14,
                                       embed_dim=1280,
                                       depth=32,
                                       num_heads=16,
                                       no_embed_class=True,
                                       init_values=1e-6,
                                       **kwargs)
    return model
