import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from captum.attr import LayerIntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import data_preprocessing as dp
import models
import config as fig

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.ndimage import gaussian_filter


def visualize_attributions(attributions, title="Sequence Position Importances"):
    """
    Display a bar chart for the averaged attribution values.
    attributions should be data in the shape of [1, seq_length].
    This assumes attributions are (1, seq_len), showing the importance of each position.
    """
    attributions_mean = attributions.mean(axis=0)
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(attributions_mean)), attributions_mean)
    plt.title(title)
    plt.xlabel("Sequence Position")
    plt.ylabel("Average Attribution")
    plt.show()


def visualize_wavy_colored_height(seq_importance, title="Sequence Position Importances (Wavy)"):
    # Ensure seq_importance is a one-dimensional numpy array
    if not isinstance(seq_importance, np.ndarray):
        seq_importance = seq_importance.detach().cpu().numpy()
    if seq_importance.ndim == 2 and seq_importance.shape[0] == 1:
        # If input is (1, seq_len), extract the first row to obtain a 1D array
        seq_importance = seq_importance[0]
    elif seq_importance.ndim != 1:
        raise ValueError("seq_importance must be a 1D array or in the form (1, seq_len).")

    # Apply Gaussian smoothing for a smoother curve
    sigma = 5
    seq_smoothed = gaussian_filter(seq_importance, sigma=sigma)

    # Use tanh scaling to prevent extreme values from dominating
    scale_factor = 1000.0
    seq_scaled = np.tanh(seq_smoothed * scale_factor)

    # Map [-1, 1] to [0, 1] and then to [0, H-1]
    H = 50  # Number of height levels, adjustable
    normalized = (seq_scaled + 1) / 2.0
    heights = normalized * (H - 1)

    # To create a seamless wavy effect, append part of the beginning data at the end
    wrap_fraction = 0.1
    wrap_len = int(len(seq_scaled) * wrap_fraction)
    extended_seq = np.concatenate([seq_scaled, seq_scaled[:wrap_len]])
    extended_heights = np.concatenate([heights, heights[:wrap_len]])
    L = len(extended_seq)

    # Construct a 2D data matrix: rows represent height levels, columns represent sequence positions
    data_2d = np.full((H, L), np.nan)
    for i in range(L):
        h = int(round(extended_heights[i]))
        data_2d[:h + 1, i] = extended_seq[i]

    # Convert NaN values to zero for interpolation and visualization
    data_2d_filled = np.nan_to_num(data_2d, nan=0.0)

    # Apply Gaussian smoothing to the 2D data for a smoother transition
    data_2d_smooth = gaussian_filter(data_2d_filled, sigma=1)

    plt.figure(figsize=(12, 4))
    plt.imshow(
        data_2d_smooth,
        cmap='RdBu',
        aspect='auto',
        interpolation='bilinear',
        origin='lower',
        vmin=-1, vmax=1
    )
    plt.colorbar(shrink=0.5)
    plt.title(title, fontsize=14)
    plt.xlabel("Sequence Position (with wrap-around)", fontsize=12)
    plt.ylabel("Height Level", fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    # Load configuration and device settings
    model_fig = fig.Config("config.ini")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define model architecture and load pre-trained parameters
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

    checkpoint_path = model_fig.ModelFolder
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    model.train()  # Use train mode to allow RNN backpropagation

    def forward_func(seq_input, pssm_input):
        return model(pssm_input, seq_input)

    # Load test dataset
    index, sequences, labels, pssm_data, sequence_lengths = dp.read_combined_h5(model_fig.Test_Database)
    vocab = {char: idx for idx, char in enumerate('ACDEFGHIKLMNPQRSTVWXY')}
    inv_vocab = {idx: char for char, idx in vocab.items()}
    test_dataset = dp.CombinedDataset(index, sequences, labels, pssm_data, sequence_lengths, vocab)

    target_index = 0
    for batch in test_dataset:
        single_sequence_item, _, _ = batch
        sequence_item_list = single_sequence_item.tolist()
        decoded_sequence = ''.join(inv_vocab[idx] for idx in sequence_item_list)
        if decoded_sequence == 'MTPSAAATGHEAADEQRLRELRGLTRQLPTGVAVVTAQDGEVAHGATVSTVSVLSQQPLRIGVSLRRGSYLTGLIRQRRVFALNVLSSRQSAVADWFANPERPRGWRQFDYVRWTAHPKAGMPVLEDALAQLHCRLTDLIPLGASDDLLVAEVLDGRGRNGRPLVNFNGRLHDVEFRGVVRVSRDQPSAVTSLE':
            break
        else:
            target_index += 1

    print("Sequence", target_index, "importance:", single_seq_importance, "\nLength:", len(single_seq_importance))
    visualize_attributions(single_seq_importance.reshape(1, -1), title="Importance of Sequence #1069")


if __name__ == "__main__":
    main()
