import os
import torch
import torch.nn as nn

import models.backbones as backbones
from models.backbones.mylora import Linear as LoraLinear
from models.backbones.mylora import DVLinear as DVLinear
from models.backbones.mylora import DoRALinear as DoRALinear
from models.backbones.mylora import MultiScaleLinear as MultiScaleLinear
from models.backbones.mylora import MoRALinear as MoRALinear


from .layers import HeadDepth
from .layers import mark_only_part_as_trainable,_make_scratch, _make_fusion_block

class DPTHead(nn.Module):
    def __init__(self, in_channels, features=128, use_bn=False, out_channels=[96, 192, 384, 768], use_clstoken=False):
        super(DPTHead, self).__init__()

        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.conv_depth_1 = HeadDepth(features)
        self.conv_depth_2 = HeadDepth(features)
        self.conv_depth_3 = HeadDepth(features)
        self.conv_depth_4 = HeadDepth(features)
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])#确定xs[0,1]判断
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        outputs = {}
        outputs[("disp", 3)] = self.sigmoid(self.conv_depth_4(path_4))
        outputs[("disp", 2)] = self.sigmoid(self.conv_depth_3(path_3))
        outputs[("disp", 1)] = self.sigmoid(self.conv_depth_2(path_2))
        outputs[("disp", 0)] = self.sigmoid(self.conv_depth_1(path_1))

        return outputs
    
class endodac(nn.Module):
    """Applies low-rank adaptation to a ViT model's image encoder.

    Args:
        backbone_size: size of pretrained Dinov2 choice from: "small", "base", "large", "giant"
        r: rank of LoRA
        image_shape: input image shape, h,w need to be multiplier of 14, default:(224,280)
        lora_layer: which layer we apply LoRA.
        lora_type: weighted
    """

    def __init__(self, 
                 backbone_size = "base", 
                 r=4,
                 image_shape=(224,280), 
                 lora_type="lora",
                 pretrained_path=None,
                 residual_block_indexes=[],
                 include_cls_token=True,
                 use_cls_token=False,
                 use_bn=False):
        super(endodac, self).__init__()

        assert r > 0

        self.r = r

        self.ranks = [4,6,8]
        self.lora_alpha = self.ranks

        self.backbone_size = backbone_size

        self.backbone = {
            "small": backbones.vits.vit_small(residual_block_indexes=residual_block_indexes,
                                              include_cls_token=include_cls_token),
            "base": backbones.vits.vit_base(residual_block_indexes=residual_block_indexes,
                                            include_cls_token=include_cls_token),
        }
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
        }
        self.intermediate_layers = {
            "small": [2, 5, 8, 11],
            "base": [2, 5, 8, 11],
        }
        self.embedding_dims = {
            "small": 384,
            "base": 768,
        }
        self.depth_head_features = {
            "small": 64,
            "base": 128,
        }
        self.depth_head_out_channels = {
            "small": [48, 96, 192, 384],
            "base": [96, 192, 384, 768],
        }
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        self.depth_head_feature = self.depth_head_features[self.backbone_size]
        self.depth_head_out_channel = self.depth_head_out_channels[self.backbone_size]
        encoder = self.backbone[self.backbone_size]

        self.image_shape = image_shape

        print("lora_type",lora_type)
        print("LoRA_r:", self.r)

        print("scale_ranks",self.ranks)

        if lora_type == "dynamic_lora":
            self._apply_dynamic_attention_lora(encoder)
        elif lora_type == "dvlora":
            self._apply_dvlora(encoder)
        elif lora_type == "dora":
            self._apply_dora(encoder)
        elif lora_type == "lora":
            self._apply_single_lora(encoder)
        elif lora_type == "mora":
            self._apply_mora(encoder)
        else:
            raise ValueError(f"Unknown lora_type: {lora_type}")


        self.encoder = encoder
        self.depth_head = DPTHead(self.embedding_dim, self.depth_head_feature, use_bn, out_channels=self.depth_head_out_channel, use_clstoken=use_cls_token)
        
        if pretrained_path is not None:
            pretrained_path = os.path.join(pretrained_path, "depth_anything_{}.pth".format(self.backbone_arch))
            pretrained_dict = torch.load(pretrained_path)
            model_dict = self.state_dict()
            self.load_state_dict(pretrained_dict, strict=False)
            print("load pretrained weight from {}\n".format(pretrained_path))

        mark_only_part_as_trainable(self.encoder)
        mark_only_part_as_trainable(self.depth_head)


    def _apply_dynamic_attention_lora(self, encoder):
        print("apply_dynamic_attention_lora")

        for t_layer_i, blk in enumerate(encoder.blocks):
            blk.mlp.fc1 = MultiScaleLinear(
                blk.mlp.fc1.in_features,
                blk.mlp.fc1.out_features,
                ranks=self.ranks,
                lora_alpha=self.lora_alpha,
                use_dynamic_fusion=True
            )
            blk.mlp.fc2 = MultiScaleLinear(
                blk.mlp.fc2.in_features,
                blk.mlp.fc2.out_features,
                ranks=self.ranks,
                lora_alpha=self.lora_alpha,
                use_dynamic_fusion=True
            )
            blk.attn.qkv = MultiScaleLinear(
                blk.attn.qkv.in_features,
                blk.attn.qkv.out_features,
                ranks=self.ranks,
                lora_alpha=self.lora_alpha,
                use_dynamic_fusion=True
            )
            blk.attn.proj = MultiScaleLinear(
                blk.attn.proj.in_features,
                blk.attn.proj.out_features,
                ranks=self.ranks,
                lora_alpha=self.lora_alpha,
                use_dynamic_fusion=True
            )

        print(" {len(encoder.blocks)} Transformer_blocks")


    def _apply_dora(self, encoder):
        print("apply_dora")
        r = self.r

        for t_layer_i, blk in enumerate(encoder.blocks):
            old_fc1 = blk.mlp.fc1
            old_fc2 = blk.mlp.fc2
            old_qkv = blk.attn.qkv
            old_proj = blk.attn.proj

            blk.mlp.fc1 = DoRALinear(
                old_fc1.in_features,
                old_fc1.out_features,
                r=r,
                lora_alpha=r
            )
            blk.mlp.fc2 = DoRALinear(
                old_fc2.in_features,
                old_fc2.out_features,
                r=r,
                lora_alpha=r
            )
            blk.attn.qkv = DoRALinear(
                old_qkv.in_features,
                old_qkv.out_features,
                r=r,
                lora_alpha=r
            )
            blk.attn.proj = DoRALinear(
                old_proj.in_features,
                old_proj.out_features,
                r=r,
                lora_alpha=r
            )

    def _apply_dvlora(self, encoder):
        print("apply_dvlora")
        r = self.r
        for t_layer_i, blk in enumerate(encoder.blocks):
            old_fc1 = blk.mlp.fc1
            old_fc2 = blk.mlp.fc2
            old_qkv = blk.attn.qkv
            old_proj = blk.attn.proj

            blk.mlp.fc1 = DVLinear(
                old_fc1.in_features,
                old_fc1.out_features,
                r=r,
                lora_alpha=r
            )
            blk.mlp.fc2 = DVLinear(
                old_fc2.in_features,
                old_fc2.out_features,
                r=r,
                lora_alpha=r
            )
            blk.attn.qkv = DVLinear(
                old_qkv.in_features,
                old_qkv.out_features,
                r=r,
                lora_alpha=r
            )
            blk.attn.proj = DVLinear(
                old_proj.in_features,
                old_proj.out_features,
                r=r,
                lora_alpha=r
            )

    def _apply_single_lora(self, encoder):
        print("single_lora")
        r = self.r

        for t_layer_i, blk in enumerate(encoder.blocks):
            old_fc1 = blk.mlp.fc1
            old_fc2 = blk.mlp.fc2
            old_qkv = blk.attn.qkv
            old_proj = blk.attn.proj

            blk.mlp.fc1 = LoraLinear(
                old_fc1.in_features,
                old_fc1.out_features,
                r=r,
                lora_alpha=r
            )
            blk.mlp.fc2 = LoraLinear(
                old_fc2.in_features,
                old_fc2.out_features,
                r=r,
                lora_alpha=r
            )
            blk.attn.qkv = LoraLinear(
                old_qkv.in_features,
                old_qkv.out_features,
                r=r,
                lora_alpha=r
            )
            blk.attn.proj = LoraLinear(
                old_proj.in_features,
                old_proj.out_features,
                r=r,
                lora_alpha=r
            )

    def _apply_mora(self, encoder):
        print("apply_MoRA")

        for t_layer_i, blk in enumerate(encoder.blocks):
            old_fc1 = blk.mlp.fc1
            old_fc2 = blk.mlp.fc2
            old_qkv = blk.attn.qkv
            old_proj = blk.attn.proj

            blk.mlp.fc1 = MoRALinear(
                old_fc1.in_features,
                old_fc1.out_features,
                mora_alpha=1,
                sparsity_ratio=0.1
            )
            blk.mlp.fc2 = MoRALinear(
                old_fc2.in_features,
                old_fc2.out_features,
                mora_alpha=1,
                sparsity_ratio=0.1
            )
            blk.attn.qkv = MoRALinear(
                old_qkv.in_features,
                old_qkv.out_features,
                mora_alpha=1,
                sparsity_ratio=0.1
            )
            blk.attn.proj = MoRALinear(
                old_proj.in_features,
                old_proj.out_features,
                mora_alpha=1,
                sparsity_ratio=0.1
            )

    def forward(self, pixel_values):
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=self.image_shape, mode="bilinear", align_corners=True)
        h, w = pixel_values.shape[-2:]
        
        features = self.encoder.get_intermediate_layers(pixel_values, 4, return_class_token=True)
        patch_h, patch_w = h // 14, w // 14

        disp = self.depth_head(features, patch_h, patch_w)

        return disp
