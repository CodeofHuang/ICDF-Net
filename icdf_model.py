# model/icdf_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork

from .DWConv import DWConv


class RevIN_2D(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        mean = x.view(x.size(0), x.size(1), -1).mean(dim=2, keepdim=True).unsqueeze(3)
        stdev = torch.sqrt(x.view(x.size(0), x.size(1), -1).var(dim=2, keepdim=True, unbiased=False) + self.eps).unsqueeze(3)
        return mean, stdev

    def forward(self, x, mode: str):
        if mode == 'norm':
            self.mean, self.stdev = self._get_statistics(x)
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.affine_weight.view(1, -1, 1, 1) + self.affine_bias.view(1, -1, 1, 1)
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.affine_bias.view(1, -1, 1, 1)) / (self.affine_weight.view(1, -1, 1, 1) + self.eps)
            x = x * self.stdev + self.mean
        return x


class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 1, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.block(x)


class Att(nn.Module):
    def __init__(self, dim=512, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, stride=1, padding=0, bias=attn_bias)
        self.oper_q = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.oper_k = nn.Sequential(
            SpatialOperation(dim),
            ChannelOperation(dim),
        )
        self.dwc = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=1)
        q = self.oper_q(q)
        k = self.oper_k(k)
        out = self.proj(self.dwc(q + k) * v)
        out = self.proj_drop(out)
        return out


class AHCM(nn.Module):

    def __init__(self, in_channels, pool_kernel=None):
        super(AHCM, self).__init__()
        self.revin = RevIN_2D(in_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.attn = Att(dim=in_channels)

    def forward(self, f1, f2):
        # 1.
        if f1.shape[2:] != f2.shape[2:]:
            H = max(f1.shape[2], f2.shape[2])
            W = max(f1.shape[3], f2.shape[3])
            f1 = F.interpolate(f1, size=(H, W), mode='bilinear', align_corners=False)
            f2 = F.interpolate(f2, size=(H, W), mode='bilinear', align_corners=False)

        # 2.
        f1_norm = self.revin(f1, 'norm')
        f2_norm = self.revin(f2, 'norm')

        # 3.
        cat_feat = torch.cat([torch.abs(f1_norm - f2_norm), f1_norm * f2_norm], dim=1)
        feat = self.fusion(cat_feat)

        # 4.
        attn_feat = self.attn(feat)
        gate = torch.sigmoid(attn_feat)

        # 5.
        f1_out = f1 + f1 * gate
        f2_out = f2 + f2 * gate

        return f1_out, f2_out


class MobileNetV2Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(MobileNetV2Backbone, self).__init__()
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        self.features = mobilenet.features
        self.c2_idx = 3
        self.c3_idx = 6
        self.c4_idx = 13
        self.c5_idx = 17
        self.channels = [24, 32, 96, 320]

    def forward(self, x):
        features_list = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.c2_idx:
                features_list.append(x)
            elif i == self.c3_idx:
                features_list.append(x)
            elif i == self.c4_idx:
                features_list.append(x)
            elif i == self.c5_idx:
                features_list.append(x)
        return features_list


class StandardFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(StandardFPN, self).__init__()
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels)

    def forward(self, features_list):
        inputs = {'feat0': features_list[0], 'feat1': features_list[1], 'feat2': features_list[2],
                  'feat3': features_list[3]}
        outputs = self.fpn(inputs)
        return outputs['feat0'], outputs['feat1'], outputs['feat2'], outputs['feat3']


class SoftThreshold(nn.Module):
    def __init__(self, initial_lambda=0.1):
        super(SoftThreshold, self).__init__()
        self.tau = nn.Parameter(torch.tensor(initial_lambda))

    def forward(self, x):
        return torch.sign(x) * torch.relu(torch.abs(x) - self.tau)


class ADMMStage(nn.Module):
    def __init__(self, in_channels):
        super(ADMMStage, self).__init__()
        self.L_update_block = DWConv(in_channels, in_channels)
        self.S_update_conv = DWConv(in_channels, in_channels)
        self.S_update_threshold = SoftThreshold()
        self.rho = nn.Parameter(torch.tensor(1.0))

    def forward(self, Z, L_k, S_k, Y_k):
        L_in = Z - S_k + (Y_k / self.rho)
        L_k_plus_1 = self.L_update_block(L_in)
        S_in = Z - L_k_plus_1 + (Y_k / self.rho)
        S_k_plus_1_conv = self.S_update_conv(S_in)
        S_k_plus_1 = self.S_update_threshold(S_k_plus_1_conv)
        Y_k_plus_1 = Y_k + self.rho * (Z - L_k_plus_1 - S_k_plus_1)
        return L_k_plus_1, S_k_plus_1, Y_k_plus_1


class DU_LRSD_Module(nn.Module):
    def __init__(self, in_channels, num_stages=5):
        super(DU_LRSD_Module, self).__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList([ADMMStage(in_channels) for _ in range(num_stages)])

    def forward(self, Z):
        L_k = torch.zeros_like(Z);
        S_k = torch.zeros_like(Z);
        Y_k = torch.zeros_like(Z)
        for stage in self.stages:
            L_k, S_k, Y_k = stage(Z, L_k, S_k, Y_k)
        return L_k, S_k


class GatedReconstructionBlock(nn.Module):
    def __init__(self, in_channels_sparse, in_channels_coarse_map, out_channels):
        super(GatedReconstructionBlock, self).__init__()
        self.conv_sparse = DWConv(in_channels_sparse, out_channels)
        self.conv_gate = nn.Sequential(
            nn.Conv2d(in_channels_coarse_map, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.smooth = DWConv(out_channels, out_channels)

    def forward(self, sparse_feat, coarse_map_up):
        F_s = self.conv_sparse(sparse_feat)
        Gate = self.conv_gate(coarse_map_up)
        F_s_filtered = F_s * Gate
        out = self.smooth(F_s_filtered + F_s)
        return out


class HSRD_Module(nn.Module):
    def __init__(self, fpn_channels=128):
        super(HSRD_Module, self).__init__()
        s_channels = fpn_channels * 2

        # Stage 5
        self.p5_to_feat = DWConv(s_channels, 128)
        self.p5_head = nn.Conv2d(128, 2, kernel_size=1)

        # Stage 4
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.recon_block4 = GatedReconstructionBlock(s_channels, 2, 128)
        self.p4_head = nn.Conv2d(128, 2, kernel_size=1)

        # Stage 3
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.recon_block3 = GatedReconstructionBlock(s_channels, 2, 128)
        self.p3_head = nn.Conv2d(128, 2, kernel_size=1)

        # Stage 2
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.recon_block2 = GatedReconstructionBlock(s_channels, 2, 64)
        self.final_head = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, S_f_multiscale):
        S_p2, S_p3, S_p4, S_p5 = S_f_multiscale

        # Stage 5
        feat_p5 = self.p5_to_feat(S_p5)
        logits_p5 = self.p5_head(feat_p5)

        # Stage 4
        logits_p5_up = self.up4(logits_p5)
        feat_p4 = self.recon_block4(S_p4, logits_p5_up)
        logits_p4 = self.p4_head(feat_p4)
        map_p4 = logits_p4 + logits_p5_up

        # Stage 3
        map_p4_up = self.up3(map_p4)
        feat_p3 = self.recon_block3(S_p3, map_p4_up)
        logits_p3 = self.p3_head(feat_p3)
        map_p3 = logits_p3 + map_p4_up

        # Stage 2
        map_p3_up = self.up2(map_p3)
        feat_p2 = self.recon_block2(S_p2, map_p3_up)
        logits_p2 = self.final_head(feat_p2)

        cm_logits_p2 = logits_p2 + map_p3_up

        cm_logits = F.interpolate(cm_logits_p2, scale_factor=4, mode='bilinear', align_corners=False)

        return cm_logits, map_p3, map_p4


class ICDF_Net(nn.Module):
    def __init__(self, opt):
        super(ICDF_Net, self).__init__()
        # 1. Backbone & FPN
        self.backbone = MobileNetV2Backbone(pretrained=True)
        self.fpn = StandardFPN(in_channels_list=self.backbone.channels, out_channels=opt.fpn_channels)

        # 2. AHCM (Alignment)
        self.ahcm_p2 = AHCM(in_channels=opt.fpn_channels)
        self.ahcm_p3 = AHCM(in_channels=opt.fpn_channels)
        self.ahcm_p4 = AHCM(in_channels=opt.fpn_channels)
        self.ahcm_p5 = AHCM(in_channels=opt.fpn_channels)

        # 3. DU-LRSD (Decomposition)
        self.decomp_p2 = DU_LRSD_Module(opt.fpn_channels, opt.num_stages)
        self.decomp_p3 = DU_LRSD_Module(opt.fpn_channels, opt.num_stages)
        self.decomp_p4 = DU_LRSD_Module(opt.fpn_channels, opt.num_stages)
        self.decomp_p5 = DU_LRSD_Module(opt.fpn_channels, opt.num_stages)

        # 4. Decoder
        self.decoder = HSRD_Module(fpn_channels=opt.fpn_channels)

    def forward(self, H1, H2):
        # --- Feature Extraction ---
        t1_feats = self.backbone(H1)
        t1_p2, t1_p3, t1_p4, t1_p5 = self.fpn(t1_feats)

        t2_feats = self.backbone(H2)
        t2_p2, t2_p3, t2_p4, t2_p5 = self.fpn(t2_feats)

        # --- 1. Alignment Process ---
        t1_p2, t2_p2 = self.ahcm_p2(t1_p2, t2_p2)
        t1_p3, t2_p3 = self.ahcm_p3(t1_p3, t2_p3)
        t1_p4, t2_p4 = self.ahcm_p4(t1_p4, t2_p4)
        t1_p5, t2_p5 = self.ahcm_p5(t1_p5, t2_p5)

        # [Batch, 128, H, W]
        diff_p2 = torch.abs(t1_p2 - t2_p2)
        diff_p3 = torch.abs(t1_p3 - t2_p3)
        diff_p4 = torch.abs(t1_p4 - t2_p4)
        diff_p5 = torch.abs(t1_p5 - t2_p5)

        # --- 2. Decomposition & Feature Fusion ---

        L_p2_1, S_p2_1 = self.decomp_p2(t1_p2);
        L_p2_2, S_p2_2 = self.decomp_p2(t2_p2)
        L_p3_1, S_p3_1 = self.decomp_p3(t1_p3);
        L_p3_2, S_p3_2 = self.decomp_p3(t2_p3)
        L_p4_1, S_p4_1 = self.decomp_p4(t1_p4);
        L_p4_2, S_p4_2 = self.decomp_p4(t2_p4)
        L_p5_1, S_p5_1 = self.decomp_p5(t1_p5);
        L_p5_2, S_p5_2 = self.decomp_p5(t2_p5)

        S_fused_p2 = S_p2_1 + S_p2_2
        S_fused_p3 = S_p3_1 + S_p3_2
        S_fused_p4 = S_p4_1 + S_p4_2
        S_fused_p5 = S_p5_1 + S_p5_2

        S_in_p2 = torch.cat([diff_p2, S_fused_p2], dim=1)
        S_in_p3 = torch.cat([diff_p3, S_fused_p3], dim=1)
        S_in_p4 = torch.cat([diff_p4, S_fused_p4], dim=1)
        S_in_p5 = torch.cat([diff_p5, S_fused_p5], dim=1)

        L_all = [(L_p2_1, L_p2_2), (L_p3_1, L_p3_2), (L_p4_1, L_p4_2), (L_p5_1, L_p5_2)]
        S_all = [(S_p2_1, S_p2_2), (S_p3_1, S_p3_2), (S_p4_1, S_p4_2), (S_p5_1, S_p5_2)]

        S_f_multiscale = (S_in_p2, S_in_p3, S_in_p4, S_in_p5)

        cm_logits, pred1, pred2 = self.decoder(S_f_multiscale)

        return cm_logits, L_all, S_all, pred1, pred2