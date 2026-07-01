#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import math
from typing import Optional, List

class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    # def train(self, mode: bool = True):
    #     def T(w):
    #         return w.T if self.fan_in_fan_out else w
    #     nn.Linear.train(self, mode)
    #     if self.merge_weights and self.merged:
    #         # Make sure that the weights are not merged
    #         if self.r > 0:
    #             self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
    #         self.merged = False

    # def eval(self):
    #     def T(w):
    #         return w.T if self.fan_in_fan_out else w
    #     nn.Linear.eval(self)
    #     if self.merge_weights and not self.merged:
    #         # Merge the weights and mark it
    #         if self.r > 0:
    #             self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
    #         self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class Embedding(nn.Embedding, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.zeros_(self.lora_A)
            nn.init.normal_(self.lora_B)

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)
        if self.merge_weights and self.merged:
            # Make sure that the weights are not merged
            if self.r > 0:
                self.weight.data -= (self.lora_B @ self.lora_A).T * self.scaling
            self.merged = False

    def eval(self):
        nn.Linear.eval(self)
        if self.merge_weights and not self.merged:
            # Merge the weights and mark it
            if self.r > 0:
                self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            result = nn.Embedding.forward(self, x)
            if self.r > 0:
                after_A = F.embedding(
                    x, self.lora_A.T, self.padding_idx, self.max_norm,
                    self.norm_type, self.scale_grad_by_freq, self.sparse
                )
                result += (after_A @ self.lora_B.T) * self.scaling
            return result
        else:
            return nn.Embedding.forward(self, x)


class DVLinear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        fan_in_fan_out: bool = False, # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        merge_weights: bool = False,
        **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.lora_U = nn.Parameter(self.weight.new_zeros(r, 1))
            self.lora_V = nn.Parameter(self.weight.new_zeros(out_features, 1))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            nn.init.kaiming_uniform_(self.lora_U, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_V, a=math.sqrt(5))
            # nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
    # def train(self, mode: bool = True):
    #     def T(w):
    #         return w.T if self.fan_in_fan_out else w
    #     nn.Linear.train(self, mode)
    #     if self.merge_weights and self.merged:
    #         # Make sure that the weights are not merged
    #         if self.r > 0:
    #             self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
    #         self.merged = False

    # def eval(self):
    #     def T(w):
    #         return w.T if self.fan_in_fan_out else w
    #     nn.Linear.eval(self)
    #     if self.merge_weights and not self.merged:
    #         # Merge the weights and mark it
    #         if self.r > 0:
    #             self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
    #         self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w
        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += (self.lora_dropout(x) @ (self.lora_A*self.lora_U).T @ (self.lora_B*self.lora_V).T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)


class DoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
        src_weight=None
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights

        # self.dora_init(src_weight)

class DoRALinear(nn.Linear, DoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = False,
            **kwargs
    ):

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.norm_init = torch.linalg.norm(self.weight, dim=1).view(-1, 1).to(self.lora_A.device)
            self.weigh_m_wdecomp = nn.Parameter(self.norm_init)
            self.weigh_m_wdecomp.requires_grad = True
            self.new_lora = nn.Parameter(torch.zeros(in_features, 1))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def cal(self, weight1, lora1):
        return weight1.T

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            if torch.linalg.norm(self.lora_B) == 0:
                self.weigh_m_wdecomp = nn.Parameter(torch.linalg.norm(self.weight, dim=1).view(-1, 1))
                self.weigh_m_wdecomp.requires_grad = True
            new_weight = self.weight + (self.lora_B @ self.lora_A) * self.scaling
            norm_scale = self.weigh_m_wdecomp.view(-1) / torch.linalg.norm(new_weight, dim=1).detach()
            org_result = F.linear(x, T(self.weight))
            dropout_x = self.lora_dropout(x)
            result = org_result + (norm_scale - 1) * F.linear(dropout_x, T(self.weight))
            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)
            result += norm_scale * (dropout_x @ self.lora_A.T @ self.lora_B.T) * self.scaling
            return result
        else:
            return F.linear(x, T(self.weight + self.new_lora.view(-1)), bias=self.bias)



class MoRALinear(nn.Linear):
    """
    MoRA: Matrix-over-Rank Adapter
    ΔW = (U ⊙ C) * scale
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mora_alpha: int = 1,
        sparsity_ratio: float = 0.1,
        mora_dropout: float = 0.,
        merge_weights: bool = False,
        **kwargs
    ):
        # Call parent init first (this will call reset_parameters())
        super().__init__(in_features, out_features, **kwargs)

        # Build mask C (register as buffer). Ensure at least one element selected.
        total = out_features * in_features
        k = max(1, int(total * sparsity_ratio))  # 防止 k==0
        mask = torch.zeros(total, dtype=torch.float32)
        mask[:k] = 1.0
        mask = mask[torch.randperm(total)].reshape(out_features, in_features)
        self.register_buffer("mora_mask", mask)

        # Create full-size trainable matrix U AFTER super().__init__
        # (cannot create before because we don't yet have device/dtype)
        self.mora_U = nn.Parameter(self.weight.new_zeros(out_features, in_features))

        # dropout
        self.mora_dropout = nn.Dropout(mora_dropout) if mora_dropout > 0 else (lambda x: x)

        # freeze original weights
        self.weight.requires_grad = False

        # scale: normalize by actual number of ones in mask (k)
        self.scale = float(mora_alpha) / float(k)

        self.merge_weights = merge_weights
        self.merged = False

        # re-run reset for subclass params initialization (safe because we've set mora_U)
        # This is optional but ensures mora_U is initialized with chosen init
        self.reset_parameters()

    def reset_parameters(self):
        # Always initialize base linear params first
        super().reset_parameters()

        # Only initialize mora_U if it exists (protect against calls from super().__init__)
        if hasattr(self, 'mora_U'):
            # Kaiming init for U
            nn.init.kaiming_uniform_(self.mora_U, a=math.sqrt(5))
        # If mora_U doesn't yet exist, we simply skip (super() already initialized weight/bias)

    def forward(self, x):
        # original linear
        out = F.linear(x, self.weight, self.bias)

        if not self.merged:
            # effective U is elementwise product with mask
            effective_U = self.mora_U * self.mora_mask
            # ΔW x
            delta = F.linear(self.mora_dropout(x), effective_U * self.scale)
            out = out + delta

        return out


class DynamicAttentionFusion(nn.Module):

    def __init__(self, num_ranks, embed_dim):
        super().__init__()
        self.num_ranks = num_ranks

        # 轻量级注意力网络
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, num_ranks * 2),
            nn.ReLU(),
            nn.Linear(num_ranks * 2, num_ranks),
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, embed_dim]
        Returns:
            fusion_weights: [batch, num_ranks]
        """
        global_feature = x.mean(dim=1)  # [batch, embed_dim]

        attention_scores = self.attention_net(global_feature)  # [batch, num_ranks]
        fusion_weights = F.softmax(attention_scores, dim=-1)  # [batch, num_ranks]

        return fusion_weights

class MultiScaleLoRALayer():
    """Multi-scale LoRA base layer"""

    def __init__(
            self,
            ranks: list,
            lora_alpha: list,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.ranks = ranks
        self.lora_alpha = lora_alpha

        if isinstance(lora_alpha, (list, tuple)):
            assert len(lora_alpha) == len(ranks), \
                "Length of lora_alpha must match ranks"
            self.lora_alpha = list(lora_alpha)
        else:
            self.lora_alpha = [lora_alpha] * len(ranks)

        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.merged = False
        self.merge_weights = merge_weights


class MultiScaleLinear(nn.Linear, MultiScaleLoRALayer):
    """Multi-scale LoRA with Dynamic Attention Fusion"""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            ranks: list = [4, 6, 8],
            lora_alpha: list = [4, 6, 8],
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            merge_weights: bool = False,
            use_dynamic_fusion: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MultiScaleLoRALayer.__init__(
            self,
            ranks=ranks,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            merge_weights=merge_weights
        )

        self.fan_in_fan_out = fan_in_fan_out
        self.use_dynamic_fusion = use_dynamic_fusion

        if len(ranks) > 0:
            self.lora_A_list = nn.ParameterList()
            self.lora_B_list = nn.ParameterList()
            self.scalings = []

            for r, alpha in zip(self.ranks, self.lora_alpha):
                lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
                lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
                self.lora_A_list.append(lora_A)
                self.lora_B_list.append(lora_B)
                self.scalings.append(alpha / r)

            if self.use_dynamic_fusion:
                self.fusion_module = DynamicAttentionFusion(
                    num_ranks=len(ranks),
                    embed_dim=in_features
                )
            else:
                self.fusion_weights = nn.Parameter(
                    torch.ones(len(ranks)) / len(ranks)
                )

            self.weight.requires_grad = False

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A_list'):
            for lora_A, lora_B in zip(self.lora_A_list, self.lora_B_list):
                nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
                nn.init.zeros_(lora_B)

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.T if self.fan_in_fan_out else w

        result = F.linear(x, T(self.weight), bias=self.bias)

        if len(self.ranks) > 0 and not self.merged:
            if self.use_dynamic_fusion:

                fusion_weights = self.fusion_module(x)  # [batch, num_ranks]

                lora_outputs = []
                for i, (lora_A, lora_B) in enumerate(zip(self.lora_A_list, self.lora_B_list)):
                    #  x @ A^T @ B^T * scaling
                    lora_out = (self.lora_dropout(x) @ lora_A.T @ lora_B.T) * self.scalings[i]
                    lora_outputs.append(lora_out)

                # lora_outputs: list of [batch, seq_len, out_features]
                # fusion_weights: [batch, num_ranks]
                for i, lora_out in enumerate(lora_outputs):
                    weight = fusion_weights[:, i].unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
                    result = result + lora_out * weight

            else:
                for i, (lora_A, lora_B) in enumerate(zip(self.lora_A_list, self.lora_B_list)):
                    lora_output = (
                                          self.lora_dropout(x) @ lora_A.T @ lora_B.T
                                  ) * self.scalings[i] * self.fusion_weights[i]
                    result += lora_output

        return result