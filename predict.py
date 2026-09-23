# predict.py
import torch
import os
from PIL import Image
import numpy as np
import random
from tqdm import tqdm
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


def predict(opt, dataset_name):
    print(f"\n{'=' * 35}\n  PREDICTING ON: {dataset_name.upper()}\n{'=' * 35}\n")

    # --- 1. ---
    opt = copy.deepcopy(opt)
    opt.dataset = dataset_name
    opt.phase = 'test'

    model_path = os.path.join(
        'vision',
        opt.name,
        dataset_name,
        'best_model.pth'
    )
    output_dir = os.path.join('predictions', opt.name, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"⚠️ Warning: Model not found at '{model_path}'. Skipping.")
        return

    # --- 2. ---
    device = torch.device(f"cuda:{opt.gpu_ids[0]}" if opt.gpu_ids and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader_wrapper = CustomDataLoader(opt)
    test_data = test_loader_wrapper.load_data()

    model = create_model(opt)

    state_dict = torch.load(model_path, map_location=device)
    model.detector.load_state_dict(state_dict)
    print(f"Model state_dict loaded from {model_path}")

    model.eval()

    # --- 3. ---
    tp, tn, fp, fn = 0, 0, 0, 0

    tbar = tqdm(test_data, ncols=100)
    with torch.no_grad():
        for data in tbar:
            img1 = data['img1'].to(device)
            img2 = data['img2'].to(device)
            label = data['cd_label'].to(device)

            pred_tensor = model.inference(img1, img2)  #

            pred_map = torch.argmax(pred_tensor.detach(), dim=1)

            tp += ((pred_map == 1) & (label == 1)).sum().item()
            tn += ((pred_map == 0) & (label == 0)).sum().item()
            fp += ((pred_map == 1) & (label == 0)).sum().item()
            fn += ((pred_map == 0) & (label == 1)).sum().item()

    # --- 4. ---
    oa, recall, precision, f1, ciou, uciou, miou, kappa = evaluate(tp, tn, fp, fn)
    num_samples = len(test_data.dataset)

    metrics_report = (
        f"--- Prediction Metrics for Dataset: {dataset_name} ---\n"
        f"Model: {model_path}\n"
        f"--------------------------------------------------\n"
        f"Total Samples: {num_samples}\n"
        f"--------------------------------------------------\n"
        f"True Positives (TP):  {tp}\n"
        f"True Negatives (TN):  {tn}\n"
        f"False Positives (FP): {fp}\n"
        f"False Negatives (FN): {fn}\n"
        f"--------------------------------------------------\n"
        f"Overall Accuracy (OA): {oa:.4f}\n"
        f"Precision:             {precision:.4f}\n"
        f"Recall:                {recall:.4f}\n"
        f"F1-Score:              {f1:.4f}\n"
        f"Change IoU (CIoU):     {ciou:.4f}\n"
        f"UnChange IoU (UCIoU):  {uciou:.4f}\n"
        f"mIoU:                  {miou:.4f}\n"
        f"Kappa:                 {kappa:.4f}\n"
        f"--------------------------------------------------\n"
    )
    print("\n" + metrics_report)

    metrics_path = os.path.join(output_dir, 'prediction_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(metrics_report)
    print(f"--> Report saved to '{metrics_path}'")


if __name__ == '__main__':
    base_opt = get_config()

    if base_opt.dataset:
        DATASET_LIST = [base_opt.dataset]
    else:
        DATASET_LIST = ['Wuhan']

    for dataset_name in DATASET_LIST:
        predict(base_opt, dataset_name)

    print("\n🎉 All datasets have been processed. Prediction finished.")