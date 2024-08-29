import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import data_preprocessing as dp
import models
import config as fig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score, confusion_matrix
import numpy as np

def compute_metrics(all_targets, all_predictions):
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, average='macro')
    recall = recall_score(all_targets, all_predictions, average='macro')
    f1 = f1_score(all_targets, all_predictions, average='macro')
    mcc = matthews_corrcoef(all_targets, all_predictions)
    try:
        auc = roc_auc_score(all_targets, all_predictions, multi_class='ovo')
    except ValueError:
        auc = float('nan')  # Prevent error when only one class is present in the samples
    try:
        auprc = average_precision_score(all_targets, all_predictions, average='macro')
    except ValueError:
        auprc = float('nan')  # Same as above, handle possible errors

    # Compute confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    tp = np.diag(cm)  # True Positives
    fp = cm.sum(axis=0) - tp  # False Positives
    fn = cm.sum(axis=1) - tp  # False Negatives
    tn = cm.sum() - (tp + fp + fn)  # True Negatives

    return accuracy, precision, recall, f1, mcc, auc, auprc, tp, tn, fp, fn

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for sequence_input, pssm_input, labels in data_loader:
            labels = labels.to(device).long()
            sequence_input = sequence_input.long().to(device)
            pssm_input = pssm_input.to(device)

            outputs = model(pssm_input, sequence_input)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_targets.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

    avg_loss = total_loss / len(data_loader)
    metrics = compute_metrics(all_targets, all_predictions)

    return avg_loss, metrics

def main():
    model_fig = fig.Config("config.ini")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    # Define model architecture
    model = models.SoluBat(
        mam_d_model=model_fig.mam_d_model,
        mam_n_layer=model_fig.mam_n_layer,
        mam_d_intermediate=model_fig.mam_d_intermediate,
        mam_vocab_size=model_fig.mam_vocab_size,
        mam_rms_norm=model_fig.mam_rms_norm,
        mam_fused_add_norm=model_fig.mam_fused_add_norm,
        mam_residual_in_fp32=model_fig.mam_residual_in_fp32,
        mam_dropout_prob=model_fig.mam_dropout_prob,
        rnn_in_channels=model_fig.rnn_in_channels,
        rnn_n_layers=model_fig.rnn_n_layers,
        rnn_conv1d_feature_size=model_fig.rnn_conv1d_feature_size,
        rnn_conv1d_kernel_size=model_fig.rnn_conv1d_kernel_size,
        rnn_avgpool1d_kernel_size=model_fig.rnn_avgpool1d_kernel_size,
        rnn_gru_hidden_size=model_fig.rnn_gru_hidden_size,
        rnn_fully_connected_layer_size=model_fig.rnn_fully_connected_layer_size,
        rnn_dropout_prob=model_fig.rnn_dropout_prob
    ).to(device)

    # Load model checkpoint
    checkpoint_path = model_fig.ModelFolder
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])

    # Load test data
    index, sequences, labels, pssm_data, sequence_lengths = dp.read_combined_h5(model_fig.Test_Database)
    vocab = {char: idx for idx, char in enumerate('ACDEFGHIKLMNPQRSTVWXY')}
    test_dataset = dp.CombinedDataset(index, sequences, labels, pssm_data, sequence_lengths, vocab)
    test_loader = DataLoader(test_dataset, batch_size=model_fig.Batch_size, shuffle=False, collate_fn=dp.collate_fn)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluate model
    test_loss, metrics = evaluate_model(model, test_loader, criterion, device)
    test_accuracy, test_precision, test_recall, test_f1, test_mcc, test_auc, test_auprc, tp, tn, fp, fn = metrics

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")
    print(f"Test MCC: {test_mcc:.4f}")
    print(f"Test AUC-ROC: {test_auc:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")

    # Display TP, TN, FP, FN for each class
    for i in range(len(tp)):
        print(f"Class {i}: TP={tp[i]}, TN={tn[i]}, FP={fp[i]}, FN={fn[i]}")

if __name__ == "__main__":
    main()
