# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
import random
import numpy as np

from domainbed.visiontransformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

__all__ = [
    'tfsvit_deit_small_patch16_224',
    'atfsvit_deit_small_patch16_224',
    'deit_small_patch16_224',
]


class TokenLevelFeatureStylizationViT(VisionTransformer):

    def __init__(self, num_layers=1, d_rate=0.5, alpha=0.1, first_layers_to_choose=0.7, **kwargs):
        super().__init__(**kwargs)
        print('--- init the TokenLevelFeatureStylizationViT')

        ### TFS information
        self.num_layers = num_layers
        self.d_rate = d_rate
        self.alpha = alpha
        self.first_layers_to_choose = int(first_layers_to_choose * kwargs['depth'])
        self.feature_stylization = FeatureStylization(alpha=self.alpha)
        print(
            f"performing Token-Level Feature Stylization on  #{num_layers} random layers (each iteration) chosen from first {self.first_layers_to_choose} layers of the network (real depth is: {kwargs['depth']}) and with drop_rate: {str(d_rate)}")

    def perform_tfs(self, x):
        """
        perform feature stylization on the input features and then randomly replace them with the rate of self.d_rate with the normal features.
        The feature stylization is performed patch-wise. So we drop "self.d_rate" of normal patches and replace them with the stylized ones.

        """
        split_point = int(self.d_rate * x.shape[1])
        x_augmented = self.feature_stylization(x)
        d0_indices, d1_indices = np.indices(x.shape[:2])
        # independetly shuffle the patch indexes
        for i in d1_indices:
            np.random.shuffle(i)

        ## debug
        # x_org = torch.clone(x)

        x[d0_indices[:, :split_point], d1_indices[:, :split_point]] = x_augmented[
            d0_indices[:, :split_point], d1_indices[:, :split_point]]

        return x

    def forward_features(self, x):

        ### randomly select #num_layers layers to perform feature stylization
        tfs_layers = np.random.choice(self.first_layers_to_choose, size=self.num_layers, replace=False) + 1

        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        ###### apply Transformer blocks + TFS
        layer_wise_tokens = []
        for l_idx, blk in enumerate(self.blocks):
            x = blk(x)
            if self.training:
                if (l_idx + 1) in tfs_layers:
                    x = self.perform_tfs(x)
            layer_wise_tokens.append(x)

        layer_wise_tokens = [self.norm(x) for x in layer_wise_tokens]
        return [(x[:, 0]) for x in layer_wise_tokens]

    def forward(self, x, return_feat=False):
        list_out = self.forward_features(x)
        features = list_out
        x = [self.head(x) for x in list_out]
        if return_feat:
            return x, features
        else:
            return x

class AttetionBasedTokenLevelFeatureStylizationViT(VisionTransformer):

    def __init__(self, num_layers=1, d_rate=0.5, alpha=0.1,
                 first_layers_to_choose=0.7, **kwargs):
        super().__init__(**kwargs)
        print('--- init the AttetionBasedTokenLevelFeatureStylizationViT')

        ### TFS information
        self.num_layers = num_layers
        self.d_rate = d_rate
        self.alpha = alpha
        self.first_layers_to_choose = int(first_layers_to_choose * kwargs['depth'])
        self.feature_stylization = FeatureStylization(alpha=self.alpha)

        print(
            f"performing AttetnionBased Token-Level Feature Stylization on  #{num_layers} random layers (each iteration) chosen from first {self.first_layers_to_choose} layers of the network (real depth is: {kwargs['depth']}) and with drop_rate: {str(d_rate)}")

    def perform_attentiontfs(self, x, attention_maps):
        """
        perform attention-based feature stylization on the input features and then randomly replace them with the rate of self.d_rate with the normal features.
        The feature stylization is performed patch-wise. So we drop "self.d_rate" of normal patches and replace them with the aumented ones.
        """
        # attention_maps shape is: (B, nh, seq, seq)
        # x shape is: (B, seq, C)

        split_point = int(self.d_rate * (x.shape[1] - 1))

        # first mean over heads and take the cls token attention maps
        attention_maps = attention_maps.mean(1)[:, 0, 1:]  # becomes (B, seq, seq) and then (B, seq-1)

        x_augmented = self.feature_stylization(x)

        d0_indices, _ = np.indices(attention_maps.shape[:2])
        d1_indices = torch.argsort(attention_maps, descending=True,
                                   dim=-1)  # att_sorted_indices  => most important (big values) first
        d1_indices = d1_indices + 1  # we won't replace the cls token

        ## debug
        # x_org = torch.clone(x)

        x[d0_indices[:, :split_point], d1_indices[:, :split_point]] = x_augmented[
            d0_indices[:, :split_point], d1_indices[:, :split_point]]


        return x

    def forward_features(self, x):

        ### randomly select #num_layers layers to perform feature stylization
        tfs_layers = np.random.choice(self.first_layers_to_choose, size=self.num_layers, replace=False) + 1

        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        ###### apply Transformer blocks + Attention-based TFS
        layer_wise_tokens = []
        for l_idx, blk in enumerate(self.blocks):
            if self.training:
                if (l_idx + 1) in tfs_layers:
                    x, att_maps = blk(x, return_attention=True)
                    x = self.perform_attentiontfs(x, att_maps)
                else:
                    x = blk(x)
            else:
                x = blk(x)
            layer_wise_tokens.append(x)

        layer_wise_tokens = [self.norm(x) for x in layer_wise_tokens]
        return [(x[:, 0]) for x in layer_wise_tokens]

    def forward(self, x, return_feat=False):
        list_out = self.forward_features(x)
        features = list_out
        x = [self.head(x) for x in list_out]
        if return_feat:
            return x, features
        else:
            return x

class FeatureStylization(nn.Module):
    """
    Most of the implementation is based on MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
      https://github.com/KaiyangZhou/mixstyle-release
    """

    def __init__(self, alpha=0.1, eps=1e-6):
        """
        Args:
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self._activated = True

    def set_activation_status(self, status=True):
        self._activated = status

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        B = x.size(0)
        # x.shape = (B, Seq, C)

        mu = x.mean(dim=1, keepdim=True)  # (B, 1, C)
        var = x.var(dim=1, keepdim=True)  # (B, 1, C)

        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix


@register_model
def tfsvit_deit_small_patch16_224(pretrained=False, **kwargs):
    model = TokenLevelFeatureStylizationViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
        print("pretrained model loaded successfully!")
    return model

@register_model
def atfsvit_deit_small_patch16_224(pretrained=False, **kwargs):
    model = AttetionBasedTokenLevelFeatureStylizationViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
        print("pretrained model loaded successfully!")
    return model

@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )
        model.load_state_dict(checkpoint["model"])
    return model
