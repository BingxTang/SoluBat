import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils
import numpy as np
import re
import config as fig
import os
import shutil
from datetime import datetime
import h5py


import torch
from torch.utils.data import Dataset


class CombinedDataset(Dataset):
    def __init__(self, indices, sequences, labels, pssm_data, sequence_lengths, vocab):
        self.vocab = vocab
        self.filtered_data = []
        # Load configuration
        model_fig = fig.Config("config.ini")
        for idx, length in enumerate(sequence_lengths):
            if model_fig.min_len <= length <= model_fig.max_len:
                pssm = pssm_data[indices[idx]]
                if not np.all(pssm == 0):
                    self.filtered_data.append((indices[idx], sequences[idx], labels[idx], pssm[:, :model_fig.rnn_in_channels], length))

    def __len__(self):
        return len(self.filtered_data)

    def __getitem__(self, idx):
        index, sequence, label, pssm, length = self.filtered_data[idx]

        # Convert sequence to integer indices
        seq_indices = [self.vocab[char] for char in sequence]
        seq_tensor = torch.tensor(seq_indices, dtype=torch.long)

        # Adjust the shape of PSSM data
        pssm_tensor = torch.tensor(pssm, dtype=torch.float32).permute(1, 0)  # Adjust to (num_features, length)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return seq_tensor, pssm_tensor, label_tensor


def read_combined_h5(file_path):
    with h5py.File(file_path, 'r') as h5file:
        index = h5file['index'][:]
        sequences = h5file['sequences'][:]
        labels = h5file['labels'][:]
        sequence_lengths = h5file['sequence_lengths'][:]

        pssm_data = {}
        pssm_group = h5file['pssm']
        for key in pssm_group.keys():
            pssm_data[int(key)] = pssm_group[key][:]

        sequences = [seq.decode('utf-8') for seq in sequences]  # Decode byte strings
        return index, sequences, labels, pssm_data, sequence_lengths


def collate_fn(batch):
    sequences, pssms, labels = zip(*batch)

    # Pad sequences
    sequences_padded = rnn_utils.pad_sequence(sequences, batch_first=True, padding_value=0)

    # Find the maximum length in PSSM features
    max_pssm_length = max([pssm.shape[1] for pssm in pssms])
    num_features = pssms[0].shape[0]

    # Pad PSSM features
    pssms_padded = []
    for pssm in pssms:
        padding = torch.zeros((num_features, max_pssm_length - pssm.shape[1]), dtype=torch.float32)
        pssm_padded = torch.cat((pssm, padding), dim=1)
        pssms_padded.append(pssm_padded)

    pssms_stacked = torch.stack(pssms_padded, dim=0)
    pssms_stacked = pssms_stacked.permute(0, 2, 1)  # Adjust shape to (batch_size, length, num_features) -> (batch_size, num_features, length)

    # Stack all label tensors together
    labels_stacked = torch.tensor(labels, dtype=torch.long)

    return sequences_padded, pssms_stacked, labels_stacked


# Resume training from checkpoints
def get_latest_checkpoint(checkpoints_dir):
    # Find all checkpoint files in the directory
    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if re.search(r'model_epoch_(\d+)', f)]

    if not checkpoint_files:
        return None  # Return None if no checkpoint files are found

    # Extract epoch number from filenames using regular expression
    checkpoint_files.sort(key=lambda f: int(re.search(r'model_epoch_(\d+)', f).group(1)))

    # Return the full path of the latest checkpoint file
    return os.path.join(checkpoints_dir, checkpoint_files[-1])


def package_checkpoints():
    # Load configuration
    model_fig = fig.Config("config.ini")

    # Define checkpoint directory and result directory
    checkpoint_dir = model_fig.Checkpoints
    result_folder = model_fig.ResultFolder

    # Create a new result directory, name includes date timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    new_result_dir = os.path.join(result_folder, f"checkpoints_{timestamp}")

    # Create result directory
    os.makedirs(new_result_dir)

    # Move all checkpoint files to the new result directory
    for filename in os.listdir(checkpoint_dir):
        file_path = os.path.join(checkpoint_dir, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            shutil.move(file_path, new_result_dir)
        elif os.path.isdir(file_path):
            shutil.move(file_path, new_result_dir)

    # Ensure the checkpoint directory is empty
    for filename in os.listdir(checkpoint_dir):
        file_path = os.path.join(checkpoint_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    # Copy config.ini file to the new result directory
    shutil.copy("config.ini", new_result_dir)

    print(f"Checkpoints have been moved to {new_result_dir}")
    print(f"Checkpoints directory {checkpoint_dir} has been cleared")
