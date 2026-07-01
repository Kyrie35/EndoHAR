from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class DynamicMultiScalePoseDecoder(nn.Module):
    """
    Pose decoder with:
      - Dynamic attention-based fusion across input frames (channel-wise attention)
      - Multi-scale feature fusion (uses last `n_scales` features from encoder)
    Outputs axisangle, translation, intermediate_feature with same shapes as original PoseDecoder.
    """

    def __init__(self,
                 num_ch_enc,
                 num_input_features,
                 num_frames_to_predict_for=None,
                 stride=1,
                 n_scales=3):
        """
        Args:
            num_ch_enc: encoder channels array, e.g. [64,64,128,256,512]
            num_input_features: number of input images (frames)
            num_frames_to_predict_for: frames to predict for (default = num_input_features - 1)
            n_scales: how many last encoder scales to use (default 3 -> uses layer4, layer3, layer2)
        """
        super(DynamicMultiScalePoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features
        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for
        self.n_scales = n_scales

        out_ch = 256
        self.squeeze_convs = nn.ModuleList()
        for i in range(self.n_scales):
            in_ch = int(self.num_ch_enc[-(i + 1)])
            self.squeeze_convs.append(nn.Conv2d(in_ch, out_ch, 1))

        # attention (GAP -> fc -> sigmoid) for each frame & scale
        self.attention_fc = nn.ModuleList()
        for i in range(self.n_scales):
            self.attention_fc.append(nn.Sequential(
                nn.Linear(out_ch, out_ch // 16, bias=True),
                nn.ReLU(inplace=True),
                nn.Linear(out_ch // 16, out_ch, bias=True),
                nn.Sigmoid()
            ))

        self.convs = OrderedDict()
        self.convs[("pose", 0)] = nn.Conv2d(self.n_scales * out_ch, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU(inplace=True)

        # register modules
        self.squeeze_convs = nn.ModuleList(self.squeeze_convs)
        self.attention_fc = nn.ModuleList(self.attention_fc)
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        """
        input_features: list of length num_input_features,
                        each element is list of encoder features (multi-scale),
                        e.g. input_features[i][s] where s indexes scale (0..4).
        This mirrors your encoder's output structure.
        """
        # collect per-frame, per-scale features
        batch = input_features[0][0].shape[0]

        # For each scale (i=0 -> last, i=1 -> second last, ...), build list of frame features
        per_scale_frame_feats = []
        for i in range(self.n_scales):
            scale_idx = -(i + 1)
            feats_this_scale = [f[scale_idx] for f in input_features]  # list length num_input_features
            per_scale_frame_feats.append(feats_this_scale)

        # squeeze (1x1 conv) for each frame & scale, and compute attention weights
        squeezed_per_scale = []
        for scale_idx, frame_feats in enumerate(per_scale_frame_feats):
            # apply squeeze conv to each frame feature at this scale
            squeezed_frames = []
            for f in frame_feats:
                s = self.squeeze_convs[scale_idx](f)  # (B, 256, H_s, W_s)
                squeezed_frames.append(s)

            # compute channel-wise attention per frame (GAP -> FC -> sigmoid)
            weights = []
            for s in squeezed_frames:
                gap = s.mean(dim=[2, 3])  # (B, C)
                w = self.attention_fc[scale_idx](gap)  # (B, C)
                w = w.view(batch, -1, 1, 1)  # (B, C, 1, 1)
                weights.append(w)

            # fused = sum(w_i * s_i) / (sum(w_i) + eps)
            eps = 1e-6
            weighted = 0.
            denom = 0.
            for w, s in zip(weights, squeezed_frames):
                weighted = weighted + w * s
                denom = denom + w
            fused = weighted / (denom + eps)  # (B, C, H_s, W_s)
            squeezed_per_scale.append(fused)

        # upsample all fused scales to the largest spatial size among selected scales
        target_h = max([f.shape[2] for f in squeezed_per_scale])
        target_w = max([f.shape[3] for f in squeezed_per_scale])
        upsampled = [F.interpolate(f, size=(target_h, target_w), mode='bilinear', align_corners=False)
                     if (f.shape[2] != target_h or f.shape[3] != target_w) else f
                     for f in squeezed_per_scale]

        cat = torch.cat(upsampled[::-1], dim=1)

        out = cat
        intermediate_feature = None
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i == 1:
                intermediate_feature = out
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)  # (B, 6*num_frames)
        out = 0.001 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation, intermediate_feature
