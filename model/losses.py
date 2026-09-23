# model/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# === Dice Loss ===
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs: [B, 2, H, W] logits
        # targets: [B, H, W] long
        inputs = F.softmax(inputs, dim=1)

        inputs = inputs[:, 1, :, :]

        targets = targets.float()

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        return 1 - dice


class ICDFLoss(nn.Module):

    def __init__(self, opt):
        super(ICDFLoss, self).__init__()
        self.alpha_rank = opt.alpha_rank
        self.lambda_ds1 = opt.lambda_ds1
        self.lambda_ds2 = opt.lambda_ds2

        self.task_loss = FocalLoss(alpha=opt.focal_alpha, gamma=opt.focal_gamma)
        self.rank_loss = nn.MSELoss()

        self.dice_loss = DiceLoss()

    def forward(self, outputs, target_long):
        D_final_logits, L_all, S_all, pred1, pred2 = outputs
        target_size = target_long.shape[1:]

        # 1. Main Task Loss (Focal + Dice)
        loss_focal = self.task_loss(D_final_logits, target_long)
        loss_dice = self.dice_loss(D_final_logits, target_long)
        loss_task = loss_focal + loss_dice

        # 2. Rank Loss
        loss_rank = 0.0
        for (L1, L2) in L_all:
            loss_rank += self.rank_loss(L1, L2)

        # 3. Deep Supervision
        pred1_up = F.interpolate(pred1, size=target_size, mode='bilinear', align_corners=False)
        pred2_up = F.interpolate(pred2, size=target_size, mode='bilinear', align_corners=False)
        loss_ds1 = self.task_loss(pred1_up, target_long)
        loss_ds2 = self.task_loss(pred2_up, target_long)

        total_loss = loss_task + \
                     self.alpha_rank * loss_rank + \
                     self.lambda_ds1 * loss_ds1 + \
                     self.lambda_ds2 * loss_ds2

        return total_loss, loss_task, loss_rank
