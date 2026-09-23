# model/create_model.py
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from .icdf_model import ICDF_Net
from .losses import ICDFLoss


def create_model(opt, train_loader=None):
    model_wrapper = ICDFModel(opt, train_loader)
    return model_wrapper


class ICDFModel(nn.Module):

    def __init__(self, opt, train_loader=None):
        super(ICDFModel, self).__init__()
        self.opt = opt
        self.device = torch.device(f"cuda:{opt.gpu_ids[0]}" if opt.gpu_ids else "cpu")

        # 1.
        self.detector = ICDF_Net(opt).to(self.device)

        # 2.
        self.loss_fn = ICDFLoss(opt).to(self.device)

        # 3.
        self.optimizer = torch.optim.AdamW(
            self.detector.parameters(),
            lr=opt.lr,
            weight_decay=opt.weight_decay
        )

        # 4.
        if train_loader:
            steps_per_epoch = len(train_loader)
            total_steps = opt.num_epochs * steps_per_epoch
            warmup_steps = opt.warmup_epochs * steps_per_epoch

            self.schedular = lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: max(1e-2,
                                           (step / warmup_steps) if step < warmup_steps else \
                                               (0.5 * (1 + torch.cos(torch.tensor(
                                                   (step - warmup_steps) / (total_steps - warmup_steps) * torch.pi))))
                                           )
            )
        else:
            self.schedular = None

    def forward(self, img1, img2, label):
        # 1.
        D_final_logits, L_all, S_all, pred1, pred2 = self.detector(img1, img2)

        # 2.
        outputs = (D_final_logits, L_all, S_all, pred1, pred2)

        # 3.
        total_loss, task, rank = self.loss_fn(outputs, label)

        loss_components = {
            'task': task.item(),
            'rank': rank.item(),
            'D_final_logits': D_final_logits
        }
        return total_loss, loss_components

    def inference(self, img1, img2):
        D_final_logits, _, _, _, _ = self.detector(img1, img2)
        return D_final_logits