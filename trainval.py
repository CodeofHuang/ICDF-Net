# trainval.py
import torch
import os
import numpy as np
import random
from tqdm import tqdm
import math
import time
import argparse

from model.create_model import create_model
from dataset import DataLoader as CustomDataLoader
from utils import evaluate
import copy

def get_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='ICDF-Net', help='name of the experiment')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0. use -1 for CPU')
    parser.add_argument('--dataroot', type=str, default='./datasets')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0, help='#threads for loading data')
    parser.add_argument('--dataset', type=str, default=None, help='Name of the dataset to process')

    parser.add_argument('--input_size', type=int, default=256)
    parser.add_argument('--fpn_channels', type=int, default=128)

    parser.add_argument('--num_stages', type=int, default=5,
                        help='Number of stages K for DU-LRSD')
    parser.add_argument('--alpha_rank', type=float, default=0.1,
                        help='Weight for L_rank (alpha) ')

    parser.add_argument('--lambda_ds1', type=float, default=0.4,
                        help='Weight for deep supervision loss')
    parser.add_argument('--lambda_ds2', type=float, default=0.2,
                        help='Weight for deep supervision loss')

    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--warmup_epochs', type=int, default=20)

    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--focal_gamma', type=int, default=2, help='gamma for Focal loss')

    opt = parser.parse_args()

    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = [int(str_id) for str_id in str_ids if int(str_id) >= 0]
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt


def train_one_epoch(model, loader, device):
    model.detector.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc="Training")

    for i, data in enumerate(progress_bar):
        img1 = data['img1'].to(device)
        img2 = data['img2'].to(device)
        label = data['cd_label'].to(device)

        loss, loss_components = model(img1, img2, label)

        model.optimizer.zero_grad()
        loss.backward()
        model.optimizer.step()
        model.schedular.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
            loss=f'{loss.item():.4f}',
            task=f"{loss_components['task']:.4f}",
            rank=f"{loss_components['rank']:.4f}",
            lr=f"{model.optimizer.param_groups[0]['lr']:.6f}"
        )

    return total_loss / len(loader)


def validate_one_epoch(model, loader, device):
    model.detector.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    total_val_loss = 0.0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validating")
        for data in progress_bar:
            img1 = data['img1'].to(device)
            img2 = data['img2'].to(device)
            label = data['cd_label'].to(device)

            loss, loss_components = model(img1, img2, label)
            total_val_loss += loss.item()

            D_final_logits = loss_components['D_final_logits']
            pred_map = torch.argmax(D_final_logits.detach(), dim=1)

            tp += ((pred_map == 1) & (label == 1)).sum().item()
            tn += ((pred_map == 0) & (label == 0)).sum().item()
            fp += ((pred_map == 1) & (label == 0)).sum().item()
            fn += ((pred_map == 0) & (label == 1)).sum().item()

    avg_val_loss = total_val_loss / len(loader)

    oa, recall, precision, f1, ciou, uciou, miou, kappa = evaluate(tp, tn, fp, fn)
    metrics = {
        'OA': oa, 'Recall': recall, 'Precision': precision, 'F1': f1,
        'CIUoU': ciou, 'UCIoU': uciou, 'mIoU': miou, 'Kappa': kappa
    }
    return metrics, avg_val_loss


if __name__ == "__main__":
    base_opt = get_config()

    if base_opt.dataset:
        DATASET_LIST = [base_opt.dataset]
    else:
        DATASET_LIST = ['Wuhan']

    for dataset_name in DATASET_LIST:
        training_start_time = time.time()
        print(f"\n{'=' * 35}\n  PROCESSING DATASET: {dataset_name.upper()}\n{'=' * 35}\n")

        opt = copy.deepcopy(base_opt)
        opt.dataset = dataset_name

        save_dir = os.path.join('vision', opt.name, dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'best_model.pth')
        best_metrics_path = os.path.join(save_dir, 'best_metrics.txt')
        train_loss_path = os.path.join(save_dir, 'train_loss.txt')
        val_loss_path = os.path.join(save_dir, 'val_loss.txt')

        opt.phase = 'train'
        train_loader_wrapper = CustomDataLoader(opt)
        train_data = train_loader_wrapper.load_data()
        val_opt = copy.deepcopy(opt)
        val_opt.phase = 'val'
        val_loader_wrapper = CustomDataLoader(val_opt)
        val_data = val_loader_wrapper.load_data()
        opt.phase = 'train'
        model = create_model(opt, train_data)
        device = model.device

        with open(train_loss_path, 'w') as f:
            f.write("Epoch,Avg_Train_Loss\n")
        with open(val_loss_path, 'w') as f:
            f.write("Epoch,Avg_Val_Loss\n")
        best_f1 = 0.0

        for epoch in range(1, opt.num_epochs + 1):
            print(f"\n==> Dataset: {opt.dataset} | Epoch: {epoch}/{opt.num_epochs}")
            train_loss = train_one_epoch(model, train_data, device)
            val_metrics, val_loss = validate_one_epoch(model, val_data, device)

            print(
                f"Validation Results - F1: {val_metrics['F1']:.4f}, mIoU: {val_metrics['mIoU']:.4f}, Val Loss: {val_loss:.4f}")

            with open(train_loss_path, 'a') as f:
                f.write(f"{epoch},{train_loss:.5f}\n")
            with open(val_loss_path, 'a') as f:
                f.write(f"{epoch},{val_loss:.5f}\n")

            if val_metrics['F1'] > best_f1:
                best_f1 = val_metrics['F1']
                print(f"  🚀 New best F1: {best_f1:.4f}. Saving model...")
                torch.save(model.detector.state_dict(), model_path)

                with open(best_metrics_path, 'w') as f_metrics:
                    f_metrics.write(f"Best F1 Score: {best_f1:.4f} achieved at Epoch {epoch}\n")
                    f_metrics.write("=" * 40 + "\n")
                    for key, value in val_metrics.items():
                        f_metrics.write(f"{key}: {value:.4f}\n")

        training_end_time = time.time()
        total_duration_s = training_end_time - training_start_time
        time_str = time.strftime("%Hh %Mm %Ss", time.gmtime(total_duration_s))
        with open(best_metrics_path, 'a') as f_metrics:
            f_metrics.write("\n" + "=" * 40 + "\n")
            f_metrics.write(f"Total Training Time: {time_str}\n")

    print("\n🎉 All datasets have been processed. Training finished.")