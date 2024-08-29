import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import optuna

import models
import config as fig
import data_preprocessing as dp
from data_preprocessing import CombinedDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

torch.cuda.empty_cache()
# Create or open log file
log_filename = 'hyper_opt_log.csv'
if not os.path.exists(log_filename):
    with open(log_filename, 'w') as log_file:
        log_file.write(
            'Trial,Learning Rate,MAM D Model,MAM N Layer,MAM D Intermediate,MAM Dropout Prob,RNN N Layers,RNN Conv1D Feature Size,RNN GRU Hidden Size,RNN Dropout Prob,Batch Size,Test Accuracy\n')


def objective(trial):
    model_fig = fig.Config("config.ini")

    if torch.cuda.is_available():
        torch.cuda.manual_seed(model_fig.Seed)
        torch.cuda.manual_seed_all(model_fig.Seed)

    # Define hyperparameters to optimize
    hyperparameters = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
        'mam_d_model': trial.suggest_categorical('mam_d_model', [16, 32, 64, 128]),
        'mam_n_layer': trial.suggest_int('mam_n_layer', 2, 6),
        'mam_d_intermediate': trial.suggest_categorical('mam_d_intermediate', [32, 64]),
        'mam_dropout_prob': trial.suggest_uniform('mam_dropout_prob', 0.2, 0.7),
        'rnn_n_layers': trial.suggest_int('rnn_n_layers', 2, 6),
        'rnn_conv1d_feature_size': trial.suggest_categorical('rnn_conv1d_feature_size', [128, 256, 512]),
        'rnn_gru_hidden_size': trial.suggest_categorical('rnn_gru_hidden_size', [128, 256, 512]),
        'rnn_dropout_prob': trial.suggest_uniform('rnn_dropout_prob', 0.2, 0.7),
        'batch_size': trial.suggest_categorical('batch_size', [8, 16])
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    index, sequences, labels, pssm_data, sequence_lengths = dp.read_combined_h5(model_fig.Train_Database)
    vocab = {char: idx for idx, char in enumerate('ACDEFGHIKLMNPQRSTVWXY')}
    train_dataset = CombinedDataset(index, sequences, labels, pssm_data, sequence_lengths, vocab)
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True,
                              collate_fn=dp.collate_fn)

    index, sequences, labels, pssm_data, sequence_lengths = dp.read_combined_h5(model_fig.Test_Database)
    test_dataset = CombinedDataset(index, sequences, labels, pssm_data, sequence_lengths, vocab)
    test_loader = DataLoader(test_dataset, batch_size=hyperparameters['batch_size'], shuffle=False,
                             collate_fn=dp.collate_fn)

    # Initialize model
    model = models.SoluBat(
        mam_d_model=hyperparameters['mam_d_model'],
        mam_n_layer=hyperparameters['mam_n_layer'],
        mam_d_intermediate=hyperparameters['mam_d_intermediate'],
        mam_vocab_size=model_fig.mam_vocab_size,
        mam_rms_norm=model_fig.mam_rms_norm,
        mam_fused_add_norm=model_fig.mam_fused_add_norm,
        mam_residual_in_fp32=model_fig.mam_residual_in_fp32,
        mam_dropout_prob=hyperparameters['mam_dropout_prob'],
        rnn_in_channels=model_fig.rnn_in_channels,
        rnn_n_layers=hyperparameters['rnn_n_layers'],
        rnn_conv1d_feature_size=hyperparameters['rnn_conv1d_feature_size'],
        rnn_conv1d_kernel_size=model_fig.rnn_conv1d_kernel_size,
        rnn_avgpool1d_kernel_size=model_fig.rnn_avgpool1d_kernel_size,
        rnn_gru_hidden_size=hyperparameters['rnn_gru_hidden_size'],
        rnn_fully_connected_layer_size=model_fig.rnn_fully_connected_layer_size,
        rnn_dropout_prob=hyperparameters['rnn_dropout_prob']
    ).to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Define learning rate adjustment strategy
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)  # Reduce learning rate every 10 epochs

    # Define checkpoint directory and result directory
    result_folder = model_fig.ResultFolder

    # Create new result directory, name includes date timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    trial_number = trial.number
    new_result_dir = os.path.join(result_folder, f"hyper_opts_{timestamp}_trial_{trial_number}")
    os.makedirs(new_result_dir, exist_ok=True)

    # Define training loop
    num_epochs = model_fig.Num_epochs
    best_test_loss = 0.0
    best_test_accuracy = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_mcc = 0.0
    best_auc = 0.0
    best_auprc = 0.0
    best_model_state = None
    best_optimizer_state = None
    best_scheduler_state = None
    best_learning_rate = None

    patience_counter = 0
    patience = 8


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

        return accuracy, precision, recall, f1, mcc, auc, auprc


    def evaluate_model(model, data_loader, criterion):
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
        accuracy, precision, recall, f1, mcc, auc, auprc = compute_metrics(all_targets, all_predictions)

        return avg_loss, accuracy, precision, recall, f1, mcc, auc, auprc

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}"):
            sequence_input, pssm_input, labels = batch
            optimizer.zero_grad()

            labels = labels.to(device).long()
            sequence_input = sequence_input.long().to(device)
            pssm_input = pssm_input.to(device)

            outputs = model(pssm_input, sequence_input)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

        # Adjust learning rate at the end of each epoch
        scheduler.step()

        test_loss, test_accuracy, test_precision, test_recall, test_f1, test_mcc, test_auc, test_auprc = evaluate_model(model, test_loader, criterion)

        if test_accuracy > best_test_accuracy:
            best_test_loss = test_loss
            best_test_accuracy = test_accuracy
            best_precision = test_precision
            best_recall = test_recall
            best_f1 = test_f1
            best_mcc = test_mcc
            best_auc = test_auc
            best_auprc = test_auprc
            best_model_state = model.state_dict()
            best_optimizer_state = optimizer.state_dict()
            best_scheduler_state = scheduler.state_dict()
            best_learning_rate = optimizer.param_groups[0]['lr']
            patience_counter = 0
        else:
            patience_counter += 1  # Increment no improvement counter

        if patience_counter >= patience:
            print(f"Early stopping triggered, training stopped at epoch {epoch + 1}.")
            break

        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test MCC: {test_mcc:.4f} Best Accuracy: {best_test_accuracy:.4f}")

    # Record hyperparameters and results
    with open(log_filename, 'a') as log_file:
        log_file.write(
            f"{trial.number},{best_learning_rate},{hyperparameters['mam_d_model']},{hyperparameters['mam_n_layer']},{hyperparameters['mam_d_intermediate']},{hyperparameters['mam_dropout_prob']},{hyperparameters['rnn_n_layers']},{hyperparameters['rnn_conv1d_feature_size']},{hyperparameters['rnn_gru_hidden_size']},{hyperparameters['rnn_dropout_prob']},{hyperparameters['batch_size']},{best_test_accuracy}\n")

    # Save the best model
    best_model_path = os.path.join(new_result_dir, f"best_model_trial_{trial_number}.pth")
    torch.save({
        "epoch": num_epochs,
        "model_state": best_model_state,
        "optimizer_state": best_optimizer_state,
        "scheduler_state": best_scheduler_state,
    }, best_model_path)

    with open(os.path.join(new_result_dir, 'basic_info.txt'), 'w') as info_file:
        info_file.write(f"Best Model saved for trial: {trial.number}\n")
        info_file.write(f"Test Accuracy: {best_test_accuracy:.4f}\n")
        info_file.write(f"Hyperparameters: {hyperparameters}\n")
        info_file.write(f"Test Loss: {best_test_loss:.4f}\n")
        info_file.write(f"Precision: {best_precision:.4f}\n")
        info_file.write(f"Recall: {best_recall:.4f}\n")
        info_file.write(f"F1 Score: {best_f1:.4f}\n")
        info_file.write(f"MCC: {best_mcc:.4f}\n")
        info_file.write(f"AUROC: {best_auc:.4f}\n")
        info_file.write(f"AUPRC: {best_auprc:.4f}\n")

    return best_test_accuracy


def save_study_log(study, log_file_path):
    with open(log_file_path, 'w') as f:
        for trial in study.trials:
            f.write(f"{trial.number},{trial.value},{trial.params}\n")


def load_study_log(log_file_path):
    study = optuna.create_study(direction='maximize')
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                trial_number = int(parts[0])
                trial_value = float(parts[1])
                trial_params = eval(','.join(parts[2:]))
                study.add_trial(optuna.trial.create_trial(
                    number=trial_number,
                    value=trial_value,
                    params=trial_params,
                    distributions={}
                ))
    return study


if __name__ == "__main__":
    study_log_file = 'optuna_study_log.txt'

    # Load or create study
    if os.path.exists(study_log_file):
        study = load_study_log(study_log_file)
        print("Loaded existing study from log file.")
    else:
        study = optuna.create_study(direction='maximize')
        print("Created new study.")

    study.optimize(objective, n_trials=50)

    # Save study log
    save_study_log(study, study_log_file)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
