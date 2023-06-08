# -------------------------------------------------------------------------------
# Modified from timm
# Copyright (c) 2015-present, Facebook, Inc. All rights reserved.
# -------------------------------------------------------------------------------

import math
from functools import partial

import torch
import torch.nn as nn
from timm.layers import PatchEmbed, Mlp, GluMlp, GatedMlp, DropPath, lecun_normal_, to_2tuple
from timm.models import named_apply

__all__ = [
    'MlpMixer',
    'mixer_s32_224',
    'mixer_s16_224',
    'mixer_b32_224',
    'mixer_b16_224',
    'mixer_b16_224_in21k',
    'mixer_l32_224',
    'mixer_l16_224',
    'mixer_l16_224_in21k',
    'mixer_b16_224_miil_in21k',
    'mixer_b16_224_miil',
    'gmixer_12_224',
    'gmixer_24_224',
    'resmlp_12_224',
    'resmlp_24_224',
    'resmlp_36_224',
    'resmlp_big_24_224',
    'resmlp_12_distilled_224',
    'resmlp_24_distilled_224',
    'resmlp_36_distilled_224',
    'resmlp_big_24_distilled_224',
    'resmlp_big_24_224_in22ft1k',
    'resmlp_12_224_dino',
    'resmlp_24_224_dino',
    'gmlp_ti16_224',
    'gmlp_s16_224',
    'gmlp_b16_224',
]


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """

    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0), mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class Affine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x):
        return torch.addcmul(self.beta, self.alpha, x)


class ResBlock(nn.Module):
    """ Residual MLP block w/ LayerScale and Affine 'norm'

    Based on: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    """

    def __init__(
            self, dim, seq_len, mlp_ratio=4, mlp_layer=Mlp, norm_layer=Affine,
            act_layer=nn.GELU, init_values=1e-4, drop=0., drop_path=0.):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm1 = norm_layer(dim)
        self.linear_tokens = nn.Linear(seq_len, seq_len)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, drop=drop)
        self.ls1 = nn.Parameter(init_values * torch.ones(dim))
        self.ls2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        x = x + self.drop_path(self.ls1 * self.linear_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.ls2 * self.mlp_channels(self.norm2(x)))
        return x


class SpatialGatingUnit(nn.Module):
    """ Spatial Gating Unit

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """

    def __init__(self, dim, seq_len, norm_layer=nn.LayerNorm):
        super().__init__()
        gate_dim = dim // 2
        self.norm = norm_layer(gate_dim)
        self.proj = nn.Linear(seq_len, seq_len)

    def init_weights(self):
        # special init for the projection gate, called as override by base model init
        nn.init.normal_(self.proj.weight, std=1e-6)
        nn.init.ones_(self.proj.bias)

    def forward(self, x):
        u, v = x.chunk(2, dim=-1)
        v = self.norm(v)
        v = self.proj(v.transpose(-1, -2))
        return u * v.transpose(-1, -2)


class SpatialGatingBlock(nn.Module):
    """ Residual Block w/ Spatial Gating

    Based on: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """

    def __init__(
            self, dim, seq_len, mlp_ratio=4, mlp_layer=GatedMlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.):
        super().__init__()
        channel_dim = int(dim * mlp_ratio)
        self.norm = norm_layer(dim)
        sgu = partial(SpatialGatingUnit, seq_len=seq_len)
        self.mlp_channels = mlp_layer(dim, channel_dim, act_layer=act_layer, gate_layer=sgu, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.mlp_channels(self.norm(x)))
        return x


class MlpMixer(nn.Module):

    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0),
            block_layer=MixerBlock,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
            global_pool='avg',
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.stem = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=norm_layer if stem_norm else None)
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim, self.stem.num_patches, mlp_ratio, mlp_layer=mlp_layer, norm_layer=norm_layer,
                act_layer=act_layer, drop=drop_rate, drop_path=drop_path_rate)
            for _ in range(num_blocks)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(nlhb=nlhb)

    @torch.jit.ignore
    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    def forward_features(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.head(x)
        return x


def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # like MLP init in vit (my original init)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()


def checkpoint_filter_fn(state_dict, model):
    """ Remap checkpoints if needed """
    if 'patch_embed.proj.weight' in state_dict:
        # Remap FB ResMlp models -> timm
        out_dict = {}
        for k, v in state_dict.items():
            k = k.replace('patch_embed.', 'stem.')
            k = k.replace('attn.', 'linear_tokens.')
            k = k.replace('mlp.', 'mlp_channels.')
            k = k.replace('gamma_', 'ls')
            if k.endswith('.alpha') or k.endswith('.beta'):
                v = v.reshape(1, 1, -1)
            out_dict[k] = v
        return out_dict
    return state_dict


def mixer_s32_224(**kwargs):
    """ Mixer-S/32 224x224
    Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model = MlpMixer(img_size=224,
                     patch_size=32,
                     num_blocks=8,
                     embed_dim=512,
                     **kwargs)
    return model


def mixer_s16_224(**kwargs):
    """ Mixer-S/16 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=8,
                     embed_dim=512,
                     **kwargs)
    return model


def mixer_b32_224(**kwargs):
    """ Mixer-B/32 224x224
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model = MlpMixer(img_size=224,
                     patch_size=32,
                     num_blocks=12,
                     embed_dim=768,
                     **kwargs)
    return model


def mixer_b16_224(**kwargs):
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224-76587d61.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=12,
                     embed_dim=768,
                     **kwargs)
    return model


def mixer_b16_224_in21k(**kwargs):
    """ Mixer-B/16 224x224. ImageNet-21k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224_in21k-617b3de2.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=12,
                     embed_dim=768,
                     **kwargs)
    return model


def mixer_l32_224(**kwargs):
    """ Mixer-L/32 224x224.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    model = MlpMixer(img_size=224,
                     patch_size=32,
                     num_blocks=24,
                     embed_dim=1024,
                     **kwargs)
    return model


def mixer_l16_224(**kwargs):
    """ Mixer-L/16 224x224. ImageNet-1k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224-92f9adc4.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=24,
                     embed_dim=1024,
                     **kwargs)
    return model


def mixer_l16_224_in21k(**kwargs):
    """ Mixer-L/16 224x224. ImageNet-21k pretrained weights.
    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224_in21k-846aa33c.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=24,
                     embed_dim=1024,
                     **kwargs)
    return model


def mixer_b16_224_miil(**kwargs):
    """ Mixer-B/16 224x224. ImageNet-21k pretrained weights.
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mixer_b16_224_miil-9229a591.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=12,
                     embed_dim=768,
                     **kwargs)
    return model


def mixer_b16_224_miil_in21k(**kwargs):
    """ Mixer-B/16 224x224. ImageNet-1k pretrained weights.
    Weights taken from: https://github.com/Alibaba-MIIL/ImageNet21K
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mixer_b16_224_miil_in21k-2a558a71.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=12,
                     embed_dim=768,
                     **kwargs)
    return model


def gmixer_12_224(**kwargs):
    """ Glu-Mixer-12 224x224
    Experiment by Ross Wightman, adding (Si)GLU to MLP-Mixer
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=12,
                     embed_dim=384,
                     mlp_ratio=(1.0, 4.0),
                     mlp_layer=GluMlp,
                     act_layer=nn.SiLU,
                     **kwargs)
    return model


def gmixer_24_224(**kwargs):
    """ Glu-Mixer-24 224x224
    Experiment by Ross Wightman, adding (Si)GLU to MLP-Mixer
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmixer_24_224_raa-7daf7ae6.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=24,
                     embed_dim=384,
                     mlp_ratio=(1.0, 4.0),
                     mlp_layer=GluMlp,
                     act_layer=nn.SiLU,
                     **kwargs)
    return model


def resmlp_12_224(**kwargs):
    """ ResMLP-12
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    https://dl.fbaipublicfiles.com/deit/resmlp_12_no_dist.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=12,
                     embed_dim=384,
                     mlp_ratio=4,
                     block_layer=ResBlock,
                     norm_layer=Affine,
                     **kwargs)
    return model


def resmlp_24_224(**kwargs):
    """ ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    https://dl.fbaipublicfiles.com/deit/resmlp_24_no_dist.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=24,
                     embed_dim=384,
                     mlp_ratio=4,
                     block_layer=partial(ResBlock, init_values=1e-5),
                     norm_layer=Affine,
                     **kwargs)
    return model


def resmlp_36_224(**kwargs):
    """ ResMLP-36
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    https://dl.fbaipublicfiles.com/deit/resmlp_36_no_dist.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=36,
                     embed_dim=384,
                     mlp_ratio=4,
                     block_layer=partial(ResBlock, init_values=1e-6),
                     norm_layer=Affine,
                     **kwargs)
    return model


def resmlp_big_24_224(**kwargs):
    """ ResMLP-B-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    https://dl.fbaipublicfiles.com/deit/resmlpB_24_no_dist.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=8,
                     num_blocks=24,
                     embed_dim=768,
                     mlp_ratio=4,
                     block_layer=partial(ResBlock, init_values=1e-6),
                     norm_layer=Affine,
                     **kwargs)
    return model


def resmlp_12_distilled_224(**kwargs):
    """ ResMLP-12
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    https://dl.fbaipublicfiles.com/deit/resmlp_12_dist.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=12,
                     embed_dim=384,
                     mlp_ratio=4,
                     block_layer=ResBlock,
                     norm_layer=Affine,
                     **kwargs)
    return model


def resmlp_24_distilled_224(**kwargs):
    """ ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    https://dl.fbaipublicfiles.com/deit/resmlp_24_dist.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=24,
                     embed_dim=384,
                     mlp_ratio=4,
                     block_layer=partial(ResBlock, init_values=1e-5),
                     norm_layer=Affine,
                     **kwargs)
    return model


def resmlp_36_distilled_224(**kwargs):
    """ ResMLP-36
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    https://dl.fbaipublicfiles.com/deit/resmlp_36_dist.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=36,
                     embed_dim=384,
                     mlp_ratio=4,
                     block_layer=partial(ResBlock, init_values=1e-6),
                     norm_layer=Affine,
                     **kwargs)
    return model


def resmlp_big_24_distilled_224(**kwargs):
    """ ResMLP-B-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    https://dl.fbaipublicfiles.com/deit/resmlpB_24_dist.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=8,
                     num_blocks=24,
                     embed_dim=768,
                     mlp_ratio=4,
                     block_layer=partial(ResBlock, init_values=1e-6),
                     norm_layer=Affine,
                     **kwargs)
    return model


def resmlp_big_24_224_in22ft1k(**kwargs):
    """ ResMLP-B-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404
    https://dl.fbaipublicfiles.com/deit/resmlpB_24_22k.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=8,
                     num_blocks=24,
                     embed_dim=768,
                     mlp_ratio=4,
                     block_layer=partial(ResBlock, init_values=1e-6),
                     norm_layer=Affine,
                     **kwargs)
    return model


def resmlp_12_224_dino(**kwargs):
    """ ResMLP-12
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404

    Model pretrained via DINO (self-supervised) - https://arxiv.org/abs/2104.14294
    https://dl.fbaipublicfiles.com/deit/resmlp_12_dino.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=12,
                     embed_dim=384,
                     mlp_ratio=4,
                     block_layer=ResBlock,
                     norm_layer=Affine,
                     **kwargs)
    return model


def resmlp_24_224_dino(**kwargs):
    """ ResMLP-24
    Paper: `ResMLP: Feedforward networks for image classification...` - https://arxiv.org/abs/2105.03404

    Model pretrained via DINO (self-supervised) - https://arxiv.org/abs/2104.14294
    https://dl.fbaipublicfiles.com/deit/resmlp_24_dino.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=24,
                     embed_dim=384,
                     mlp_ratio=4,
                     block_layer=partial(ResBlock, init_values=1e-5),
                     norm_layer=Affine,
                     **kwargs)
    return model


def gmlp_ti16_224(**kwargs):
    """ gMLP-Tiny
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=30,
                     embed_dim=128,
                     mlp_ratio=6,
                     block_layer=SpatialGatingBlock,
                     mlp_layer=GatedMlp,
                     **kwargs)
    return model


def gmlp_s16_224(**kwargs):
    """ gMLP-Small
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/gmlp_s16_224_raa-10536d42.pth
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=30,
                     embed_dim=256,
                     mlp_ratio=6,
                     block_layer=SpatialGatingBlock,
                     mlp_layer=GatedMlp,
                     **kwargs)
    return model


def gmlp_b16_224(**kwargs):
    """ gMLP-Base
    Paper: `Pay Attention to MLPs` - https://arxiv.org/abs/2105.08050
    """
    model = MlpMixer(img_size=224,
                     patch_size=16,
                     num_blocks=30,
                     embed_dim=512,
                     mlp_ratio=6,
                     block_layer=SpatialGatingBlock,
                     mlp_layer=GatedMlp,
                     **kwargs)
    return model
