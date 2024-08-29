import models
import config as fig
import data_preprocessing as dp
import pandas as pd
import numpy as np
from data_preprocessing import CombinedDataset
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import configparser
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader, Dataset
import time
from datetime import datetime
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score

torch.cuda.empty_cache()
model_fig = fig.Config("config.ini")
# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.cuda.manual_seed(model_fig.Seed)
    torch.cuda.manual_seed_all(model_fig.Seed)

index, sequences, labels, pssm_data, sequence_lengths = dp.read_combined_h5(model_fig.Train_Database)
vocab = {char: idx for idx, char in enumerate('ACDEFGHIKLMNPQRSTVWXY')}

train_dataset = CombinedDataset(index, sequences, labels, pssm_data, sequence_lengths, vocab)
train_loader = DataLoader(train_dataset, batch_size=model_fig.Batch_size, shuffle=True, collate_fn=dp.collate_fn)

index, sequences, labels, pssm_data, sequence_lengths = dp.read_combined_h5(model_fig.Test_Database)
test_dataset = CombinedDataset(index, sequences, labels, pssm_data, sequence_lengths, vocab)
test_loader = DataLoader(test_dataset, batch_size=model_fig.Batch_size, shuffle=False, collate_fn=dp.collate_fn)

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

# Ensure the loss function is suitable for the current task
optimizer = optim.Adam(model.parameters(), lr=model_fig.learning_rate)
criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss

# Define training loop
num_epochs = model_fig.Num_epochs  # Number of training epochs

print("Data loaded from:", model_fig.Train_Database)
print("Using device:", device)

# Setup TensorBoard
writer = SummaryWriter(model_fig.Logdir)
# Define step interval for saving the model
save_interval = 5  # Save the model every 5 epochs
# Define learning rate adjustment strategy
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)  # Reduce learning rate every 10 epochs

# Initialize start epoch
start_epoch = 0

# Number of epochs without improvement
patience = 100

# Track the number of epochs without improvement
patience_counter = 0

# Initialize the best test accuracy
best_test_accuracy = 0.0

# Define checkpoint directory
checkpoint_dir = model_fig.Checkpoints

# Get the latest checkpoint path
latest_checkpoint_path = dp.get_latest_checkpoint(checkpoint_dir)

# If the latest checkpoint is found, load it
if latest_checkpoint_path:
    checkpoint = torch.load(latest_checkpoint_path)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Loaded the latest checkpoint: {latest_checkpoint_path}, starting training from epoch {start_epoch}.")
else:
    print("No checkpoint found, starting training from epoch 1.")

# Create or open log file
log_filename = os.path.join(checkpoint_dir, datetime.now().strftime('%Y-%m-%d') + '_training_log.txt')
log_file_mode = 'a' if start_epoch > 0 else 'w'
if os.path.exists(log_filename):
    log_data = pd.read_csv(log_filename)
    if not log_data.empty:
        best_test_accuracy = log_data['Test Accuracy'].max()
        print(f"Read the highest test accuracy from the log file: {best_test_accuracy:.4f}")
else:
    with open(log_filename, log_file_mode) as log_file:
        if start_epoch == 0:
            log_file.write('Epoch,Training Time,Training Loss,Test Loss,Test Accuracy,Best Test Accuracy\n')


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


for epoch in range(start_epoch, num_epochs):
    model.train()  # Set model to training mode
    total_loss = 0  # Accumulate loss
    start_time = time.time()

    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}"):
        # Get input data and labels
        sequence_input, pssm_input, labels = batch
        optimizer.zero_grad()

        # Move inputs and labels to the specified device
        labels = labels.to(device).long()
        sequence_input = sequence_input.long().to(device)
        pssm_input = pssm_input.to(device)

        # Forward pass
        outputs = model(pssm_input, sequence_input)  # Model output

        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        # Record loss
        total_loss += loss.item()

        # Log loss to TensorBoard
        writer.add_scalar("Training Loss", loss.item(), epoch * len(train_loader) + i)

    # Output average loss for each epoch
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Scheduler: {scheduler.optimizer}")

    # Adjust learning rate at the end of each epoch
    scheduler.step()

    # Calculate training duration
    epoch_time = time.time() - start_time

    # Log average loss per epoch to TensorBoard
    writer.add_scalar("Average Loss per Epoch", avg_loss, epoch)

    # Evaluate model performance on the test set
    test_loss, test_accuracy, test_precision, test_recall, test_f1, test_mcc, test_auc, test_auprc = evaluate_model(model, test_loader, criterion)

    # Check for new best performance
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        patience_counter = 0  # Reset no improvement counter
        # Save the best model
        best_checkpoint_path = checkpoint_dir + "/" + f"best_model.pth"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, best_checkpoint_path)
    else:
        patience_counter += 1  # Increment no improvement counter

    # Check if early stopping threshold is reached
    if patience_counter >= patience:
        print(f"Early stopping triggered, training stopped at epoch {epoch + 1}.")
        break

    writer.add_scalar("Test Loss", test_loss, epoch)
    writer.add_scalar("Test Accuracy", test_accuracy, epoch)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Recall: {test_recall:.4f}, Test F1: {test_f1:.4f}, Test MCC: {test_mcc:.4f} Best Accuracy: {best_test_accuracy:.4f}")

    # Log to file
    with open(log_filename, 'a') as log_file:
        log_file.write(f"{epoch + 1},{epoch_time:.2f},{avg_loss:.4f},{test_loss:.4f},{test_accuracy:.4f},{best_test_accuracy:.4f}\n")

    # Save checkpoint
    if (epoch + 1) % save_interval == 0 or best_test_accuracy == test_accuracy:
        checkpoint_path = checkpoint_dir + "/" + f"model_epoch_{epoch + 1}.pth"
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        }, checkpoint_path)
        print(f"Checkpoint saved to: {checkpoint_path}")

dp.package_checkpoints()
writer.close()  # Close TensorBoard
