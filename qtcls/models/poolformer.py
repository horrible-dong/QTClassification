# --------------------------------
# Modified from timm by QIU Tian
# --------------------------------

# -------------------------------------------------------------------------------
# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------------

__all__ = [
    'PoolFormer',
    'poolformer_s12',
    'poolformer_s24',
    'poolformer_s36',
    'poolformer_m36',
    'poolformer_m48'
]

import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_, to_2tuple, ConvMlp, GroupNorm1


class PatchEmbed(nn.Module):
    """ Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, in_chs=3, embed_dim=768, patch_size=16, stride=16, padding=0, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chs, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class PoolFormerBlock(nn.Module):
    """
    Args:
        dim: embedding dim
        pool_size: pooling size
        mlp_ratio: mlp expansion ratio
        act_layer: activation
        norm_layer: normalization
        drop: dropout rate
        drop path: Stochastic Depth, refer to https://arxiv.org/abs/1603.09382
        use_layer_scale, --layer_scale_init_value: LayerScale, refer to https://arxiv.org/abs/2103.17239
    """

    def __init__(
            self, dim, pool_size=3, mlp_ratio=4.,
            act_layer=nn.GELU, norm_layer=GroupNorm1,
            drop=0., drop_path=0., layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = ConvMlp(dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if layer_scale_init_value:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.layer_scale_1 = None
            self.layer_scale_2 = None

    def forward(self, x):
        if self.layer_scale_1 is not None:
            x = x + self.drop_path1(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path2(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


def basic_blocks(
        dim, index, layers,
        pool_size=3, mlp_ratio=4.,
        act_layer=nn.GELU, norm_layer=GroupNorm1,
        drop_rate=.0, drop_path_rate=0.,
        layer_scale_init_value=1e-5,
):
    """ generate PoolFormer blocks for a stage """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PoolFormerBlock(
            dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            layer_scale_init_value=layer_scale_init_value,
        ))
    blocks = nn.Sequential(*blocks)
    return blocks


class PoolFormer(nn.Module):
    """ PoolFormer
    """

    def __init__(
            self,
            layers,
            embed_dims=(64, 128, 320, 512),
            mlp_ratios=(4, 4, 4, 4),
            downsamples=(True, True, True, True),
            pool_size=3,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            norm_layer=GroupNorm1,
            act_layer=nn.GELU,
            in_patch_size=7,
            in_stride=4,
            in_pad=2,
            down_patch_size=3,
            down_stride=2,
            down_pad=1,
            drop_rate=0., drop_path_rate=0.,
            layer_scale_init_value=1e-5,
            **kwargs):

        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = embed_dims[-1]

        self.patch_embed = PatchEmbed(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chs=in_chans, embed_dim=embed_dims[0])

        # set the main block in network
        network = []
        for i in range(len(layers)):
            network.append(basic_blocks(
                embed_dims[i], i, layers,
                pool_size=pool_size, mlp_ratio=mlp_ratios[i],
                act_layer=act_layer, norm_layer=norm_layer,
                drop_rate=drop_rate, drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value)
            )
            if i < len(layers) - 1 and (downsamples[i] or embed_dims[i] != embed_dims[i + 1]):
                # downsampling between stages
                network.append(PatchEmbed(
                    in_chs=embed_dims[i], embed_dim=embed_dims[i + 1],
                    patch_size=down_patch_size, stride=down_stride, padding=down_pad)
                )

        self.network = nn.Sequential(*network)
        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    # init for classification
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.network(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == 'avg':
            x = x.mean([-2, -1])
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def poolformer_s12(**kwargs):
    """
    PoolFormer-S12 model, Params: 12M
    https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s12.pth.tar
    """
    model = PoolFormer(layers=(2, 2, 6, 2),
                       **kwargs)
    return model


def poolformer_s24(**kwargs):
    """
    PoolFormer-S24 model, Params: 21M
    https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s24.pth.tar
    """
    model = PoolFormer(layers=(4, 4, 12, 4),
                       **kwargs)
    return model


def poolformer_s36(**kwargs):
    """
    PoolFormer-S36 model, Params: 31M
    https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_s36.pth.tar
    """
    model = PoolFormer(layers=(6, 6, 18, 6),
                       layer_scale_init_value=1e-6,
                       **kwargs)
    return model


def poolformer_m36(**kwargs):
    """
    PoolFormer-M36 model, Params: 56M
    https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m36.pth.tar
    """
    model = PoolFormer(layers=(6, 6, 18, 6),
                       embed_dims=(96, 192, 384, 768),
                       layer_scale_init_value=1e-6,
                       **kwargs)
    return model


def poolformer_m48(**kwargs):
    """
    PoolFormer-M48 model, Params: 73M
    https://github.com/sail-sg/poolformer/releases/download/v1.0/poolformer_m48.pth.tar
    """
    model = PoolFormer(layers=(8, 8, 24, 8),
                       embed_dims=(96, 192, 384, 768),
                       layer_scale_init_value=1e-6,
                       **kwargs)
    return model
