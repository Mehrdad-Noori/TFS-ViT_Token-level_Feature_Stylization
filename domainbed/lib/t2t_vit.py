# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
import random

import torch
import torch.nn as nn

from timm.models.helpers import load_pretrained
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
from .token_transformer import Token_transformer
from .token_performer import Token_performer
from .transformer_block import Block, get_sinusoid_encoding


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'T2t_vit_7': _cfg(),
    'T2t_vit_10': _cfg(),
    'T2t_vit_12': _cfg(),
    'T2t_vit_14': _cfg(),
    'T2t_vit_19': _cfg(),
    'T2t_vit_24': _cfg(),
    'T2t_vit_t_14': _cfg(),
    'T2t_vit_t_19': _cfg(),
    'T2t_vit_t_24': _cfg(),
    'T2t_vit_14_resnext': _cfg(),
    'T2t_vit_14_wide': _cfg(),
}


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """

    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        if tokens_type == 'transformer':
            print('adopt transformer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'performer':
            print('adopt performer encoder for tokens-to-token')
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            # self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            # self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(dim=in_chans * 7 * 7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim * 3 * 3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == 'convolution':  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print('adopt convolution layers for tokens-to-token')
            self.soft_split0 = nn.Conv2d(3, token_dim, kernel_size=(7, 7), stride=(4, 4),
                                         padding=(2, 2))  # the 1st convolution
            self.soft_split1 = nn.Conv2d(token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2),
                                         padding=(1, 1))  # the 2nd convolution
            self.project = nn.Conv2d(token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2),
                                     padding=(1, 1))  # the 3rd convolution

        self.num_patches = (img_size // (4 * 2 * 2)) * (
                img_size // (4 * 2 * 2))  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x


class T2T_ViT(nn.Module):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
            img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim),
                                      requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        layer_wise_tokens = []

        for blk in self.blocks:
            x = blk(x)
            layer_wise_tokens.append(x)

        layer_wise_tokens = [self.norm(x) for x in layer_wise_tokens]
        return [(x[:, 0]) for x in layer_wise_tokens]

    def forward(self, x):
        list_out = self.forward_features(x)
        x = [self.head(x) for x in list_out]
        return x


class TokenLevelFeatureStylization_T2T_VIT(T2T_ViT):
    def __init__(self, num_layers=1, d_rate=0.5, alpha=0.1, first_layers_to_choose=0.7, **kwargs):
        super().__init__(**kwargs)
        print('--- init the TokenLevelFeatureStylization_T2T_VIT')

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

        # independently shuffle the patch indexes
        for i in d1_indices:
            np.random.shuffle(i)

        x[d0_indices[:, :split_point], d1_indices[:, :split_point]] = x_augmented[
            d0_indices[:, :split_point], d1_indices[:, :split_point]]

        return x

    def forward_features(self, x):

        # randomly select #num_layers layers to perform feature stylization
        tfs_layers = np.random.choice(self.first_layers_to_choose, size=self.num_layers, replace=False) + 1

        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        ###### apply Transformer blocks + TFS
        layer_wise_tokens = []
        for blk_idx, blk in enumerate(self.blocks):
            x = blk(x)
            if self.training:
                if (blk_idx + 1) in tfs_layers:
                    x = self.perform_tfs(x)
            layer_wise_tokens.append(x)

        layer_wise_tokens = [self.norm(x) for x in layer_wise_tokens]
        return [(x[:, 0]) for x in layer_wise_tokens]

    def forward(self, x):
        list_out = self.forward_features(x)
        x = [self.head(x) for x in list_out]
        return x


class AttentionBasedTokenLevelFeatureStylization_T2T_VIT(T2T_ViT):
    def __init__(self, num_layers=1, d_rate=0.5, alpha=0.1, first_layers_to_choose=0.7, **kwargs):
        super().__init__(**kwargs)
        print('+++ init the AttentionBasedTokenLevelFeatureStylization_T2T_VIT')
        self.num_layers = num_layers
        self.d_rate = d_rate
        self.alpha = alpha
        self.first_layers_to_choose = int(first_layers_to_choose * kwargs['depth'])
        self.feature_stylization = FeatureStylization(alpha=self.alpha)

        print(f"performing AttetnionBased Token-Level Feature Stylization on  #{num_layers} random layers (each iteration) chosen from first {self.first_layers_to_choose} layers of the network (real depth is: {kwargs['depth']}) and with drop_rate: {str(d_rate)}")

    def perform_attentiontfs(self, x, attention_maps):
        """
        perform attention-based feature stylization on the input features and then randomly replace them with the rate of self.d_rate with the normal features.
        The feature stylization is performed patch-wise. So we drop "self.d_rate" of normal patches and replace them with the aumented ones.
        """
        # attention_maps shape is: (B, nh, seq, seq)
        # x shape is: (B, seq, C)

        split_point = int(self.d_rate * (x.shape[1]-1))

        # first mean over heads and take the cls token attention maps
        attention_maps = attention_maps.mean(1)[:, 0, 1:]  # becomes (B, seq, seq) and then (B, seq-1)

        x_augmented = self.feature_stylization(x)

        d0_indices, _ = np.indices(attention_maps.shape[:2])

        d1_indices = torch.argsort(attention_maps, descending=True, dim=-1)  # att_sorted_indices  => most important (big values) first
        d1_indices = d1_indices + 1 # we won't replace the cls token

        ## debug
        # x_org = torch.clone(x)

        x[d0_indices[:, :split_point], d1_indices[:, :split_point]] = x_augmented[d0_indices[:, :split_point], d1_indices[:, :split_point]]

        return x


    def forward_features(self, x):

        # randomly select #num_layers layers to perform feature stylization
        tfs_layers = np.random.choice(self.first_layers_to_choose, size=self.num_layers, replace=False) + 1

        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        ###### apply Transformer blocks + AttentionBased TFS
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

    def forward(self, x):
        list_out = self.forward_features(x)
        x = [self.head(x) for x in list_out]
        return x


class T2T_ViT_RB(nn.Module):
    def __init__(self, img_size=224, tokens_type='performer', in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, token_dim=64):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
            img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim)
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim),
                                      requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        layer_wise_tokens = []

        for blk in self.blocks:
            x = blk(x)
            layer_wise_tokens.append(x)

        layer_wise_tokens = [self.norm(x) for x in layer_wise_tokens]
        return [(x[:, 0]) for x in layer_wise_tokens]

    def forward(self, x):
        list_out = self.forward_features(x)
        x = [self.head(x) for x in list_out]
        block_number = random.randint(0, len(self.blocks) - 1)
        x1 = x[-1]
        x_random_block = x[block_number]
        if self.training:
            return x1, x_random_block
        else:
            return x1


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
def t2t_vit_t_14_RB(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT_RB(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    if pretrained:
        model.load_state_dict(
            torch.load("/home/computervision1/DG_new_idea/domainbed/pretrained/81.7_T2T_ViTt_14.pth",
                       map_location="cpu"), strict=True)
    return model


@register_model
def t2t_vit_7(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=7, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_7']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_10(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=10, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_10']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_12(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 256 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=256, depth=12, num_heads=4, mlp_ratio=2., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_12']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_14(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_19(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_24(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_t_14(**kwargs):  # adopt transformers for tokens to token
    model = T2T_ViT(tokens_type='transformer', embed_dim=384, depth=14, num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    return model

@register_model
def atfsvit_t2t_vit_t_14(num_layers=1, d_rate=0.5, alpha=0.1, first_layers_to_choose=0.7, **kwargs):
    model = AttentionBasedTokenLevelFeatureStylization_T2T_VIT(num_layers, d_rate, alpha,
                                   first_layers_to_choose, tokens_type='transformer', embed_dim=384, depth=14,
                                   num_heads=6, mlp_ratio=3.,  **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    return model

@register_model
def tfsvit_t2t_vit_t_14(num_layers=1, d_rate=0.5, alpha=0.1, first_layers_to_choose=0.7, **kwargs):
    model = TokenLevelFeatureStylization_T2T_VIT(num_layers, d_rate, alpha,
                                   first_layers_to_choose, tokens_type='transformer', embed_dim=384, depth=14,
                                   num_heads=6, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_14']
    return model


@register_model
def t2t_vit_t_19(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 448 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=448, depth=19, num_heads=7, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_19']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_t_24(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='transformer', embed_dim=512, depth=24, num_heads=8, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_t_24']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


# rexnext and wide structure
@register_model
def t2t_vit_14_resnext(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=384, depth=14, num_heads=32, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_resnext']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model


@register_model
def t2t_vit_14_wide(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault('qk_scale', 512 ** -0.5)
    model = T2T_ViT(tokens_type='performer', embed_dim=768, depth=4, num_heads=12, mlp_ratio=3., **kwargs)
    model.default_cfg = default_cfgs['T2t_vit_14_wide']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3))
    return model
