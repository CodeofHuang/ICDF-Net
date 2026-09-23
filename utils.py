import torch
import torch.utils.data.dataloader


def confusion_matrix(true_value, output_data):
    image_size = true_value.shape[2]
    true_positive_sum, true_negative_sum, false_positive_sum, false_negative_sum = 0, 0, 0, 0
    output_data = torch.sigmoid(output_data) > 0.5
    true_value = true_value.bool()

    batch_size = true_value.shape[0]
    for i in range(batch_size):
        # True Positive (TP)
        true_positive = (output_data[i] & true_value[i]).sum().item()
        # True Negative (TN)
        true_negative = (~output_data[i] & ~true_value[i]).sum().item()
        # False Positive (FP)
        false_positive = (output_data[i] & ~true_value[i]).sum().item()
        # False Negative (FN)
        false_negative = (~output_data[i] & true_value[i]).sum().item()

        true_positive_sum += true_positive
        true_negative_sum += true_negative
        false_positive_sum += false_positive
        false_negative_sum += false_negative

    return true_positive_sum, true_negative_sum, false_positive_sum, false_negative_sum


def evaluate(tp, tn, fp, fn):
    tp, tn, fp, fn = float(tp), float(tn), float(fp), float(fn)
    pcc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    # Recall (Sensitivity)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    # F1-Score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    # Change Intersection over Union (CIOU)
    ciou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    # Unchanged Intersection over Union (UCIOU)
    uciou = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
    # Mean Intersection over Union (MIOU)
    miou = (ciou + uciou) / 2

    # Kappa Coefficient
    epsilon = 1e-6
    total_pixels = tp + tn + fp + fn
    p_e = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (total_pixels ** 2 + epsilon)
    kappa = (pcc - p_e) / (1 - p_e + epsilon)

    return pcc, recall, precision, f1, ciou, uciou, miou, kappa