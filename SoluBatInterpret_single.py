import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import LayerIntegratedGradients
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from scipy.ndimage import gaussian_filter

import models
import config as fig

#########################
# 1. Read local GFP_seq and GFP_pssm
#########################
def read_gfp_seq(seq_file_path):
    """
    Assume the format of GFP_seq.txt is as follows (one line only):
    MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK 1
    where the last "1" indicates the label (soluble: 1 / insoluble: 0).
    """
    with open(seq_file_path, "r") as f:
        line = f.readline().strip()
        # Split by spaces or tabs, assuming the last field is the label and the rest is the sequence
        parts = line.split()
        seq = "".join(parts[:-1])
        label = int(parts[-1])
    return seq, label


def read_gfp_pssm(pssm_file_path):
    """
    Read the PSSM matrix from the specified path, assuming each line is in the following format:

    1 M  -3  -4  -4  -5  ...  14  1.60  0.43
    2 S   2  -1   0   1  ...  37  0.56  0.23
    ...

    - parts[0] => Row number
    - parts[1] => Amino acid character
    - parts[2:44] => 42 values (20 score columns + 20 percentage columns + 2 info columns)

    Returns a numpy array of shape (L, 42), where L is the sequence length.
    """
    pssm_data = []
    with open(pssm_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line_strip = line.strip()
            # Skip empty lines or header lines
            if not line_strip:
                continue
            # Check if the line starts with a digit (indicating amino acid position)
            parts = line_strip.split()
            if parts[0].isdigit():
                # This line should contain PSSM data
                if len(parts) < 44:
                    continue
                try:
                    numeric_vals = list(map(float, parts[2:44]))  # [2:44] => 42 columns
                except ValueError:
                    continue
                if len(numeric_vals) == 42:
                    pssm_data.append(numeric_vals)

    pssm_data = np.array(pssm_data, dtype=np.float32)  # shape: (L, 42)
    return pssm_data


#########################
# 2. Visualization functions
#########################
def visualize_attributions(attributions, title="Sequence Position Importances"):
    """
    Visualize the average attribution values in a bar chart.
    attributions should be in the shape of [1, seq_length],
    here we assume attributions is (1, seq_len).
    """
    attributions_mean = attributions.mean(axis=0)
    plt.figure(figsize=(10,4))
    plt.bar(range(len(attributions_mean)), attributions_mean)
    plt.title(title)
    plt.xlabel("Sequence Position")
    plt.ylabel("Average Attribution")
    plt.show()


def visualize_wavy_colored_height(seq_importance, title="Sequence Position Importances (Wavy)"):
    """
    Visualize the sequence position importance in a wavy pattern.
    seq_importance: a 1D numpy array or a numpy array of shape (1, seq_len)
    """
    if not isinstance(seq_importance, np.ndarray):
        seq_importance = seq_importance.detach().cpu().numpy()
    if seq_importance.ndim == 2 and seq_importance.shape[0] == 1:
        seq_importance = seq_importance[0]
    elif seq_importance.ndim != 1:
        raise ValueError("seq_importance must be a 1D or (1, seq_len) array.")

    # Apply Gaussian smoothing to make the curve smoother
    sigma = 5
    seq_smoothed = gaussian_filter(seq_importance, sigma=sigma)

    # Scale using tanh to avoid extreme values
    scale_factor = 1000.0
    seq_scaled = np.tanh(seq_smoothed * scale_factor)

    # Map [-1, 1] to [0, 1] and then to [0, H-1]
    H = 50  # Number of height levels, adjustable
    normalized = (seq_scaled + 1) / 2.0
    heights = normalized * (H - 1)

    # To create a "wave" wrap at the ends, append part of the beginning data to the end
    wrap_fraction = 0.1
    wrap_len = int(len(seq_scaled)*wrap_fraction)
    extended_seq = np.concatenate([seq_scaled, seq_scaled[:wrap_len]])
    extended_heights = np.concatenate([heights, heights[:wrap_len]])
    L = len(extended_seq)

    data_2d = np.full((H, L), np.nan)
    for i in range(L):
        h = int(round(extended_heights[i]))
        data_2d[:h+1, i] = extended_seq[i]

    data_2d_filled = np.nan_to_num(data_2d, nan=0.0)
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


#########################
# 3. Main process
#########################
def main():
    # Load configuration and device
    model_fig = fig.Config("config.ini")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read a single GFP sequence and label
    seq, label = read_gfp_seq("GFP_seq.txt")
    pssm_array = read_gfp_pssm("GFP_seq.pssm")  # Your PSSM file name
    print("pssm length: ", len(pssm_array), len(pssm_array[0]))

    # Define mapping from amino acid to index
    vocab = {char: idx for idx, char in enumerate('ACDEFGHIKLMNPQRSTVWXY')}

    # Convert sequence to indices
    seq_indices = [vocab[aa] for aa in seq]
    sequence_input = torch.tensor(seq_indices, dtype=torch.long).unsqueeze(0).to(device)  # [1, seq_len]

    # Convert pssm_array to tensor shape [1, seq_len, 42], then permute to [1, 42, seq_len]
    pssm_tensor = torch.tensor(pssm_array, dtype=torch.float32).unsqueeze(0)  # => [1, seq_len, 42]
    pssm_tensor = pssm_tensor.permute(0, 2, 1).to(device)                    # => [1, 42, seq_len]

    # Define model structure and load trained model parameters
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    # Switch to train mode to allow RNN backpropagation (or change to eval mode)
    model.train()

    ###########################
    #  3.1 Make inference (predict soluble/insoluble)
    ###########################
    with torch.no_grad():
        output = model(pssm_tensor, sequence_input)  # shape: [1, num_classes=2]
        pred_label = torch.argmax(output, dim=1).item()
        if pred_label == 1:
            print("【Inference result】：The model predicts the sequence is 'Soluble'")
        else:
            print("【Inference result】：The model predicts the sequence is 'Insoluble'")

    ###########################
    #  3.2 Perform interpretability analysis
    ###########################
    # Define forward_func, needed by Captum
    def forward_func(seq_input, pssm_input):
        return model(pssm_input, seq_input)

    # Choose the target class to analyze, assuming 1 is "Soluble" class
    target_class = 1

    # Create LayerIntegratedGradients
    lig = LayerIntegratedGradients(forward_func, model.mamaba.embedding)

    # Baseline is a zero sequence tensor
    baseline_seq = torch.zeros_like(sequence_input)

    model.zero_grad()
    seq_attributions = lig.attribute(
        sequence_input,
        baselines=baseline_seq,
        additional_forward_args=(pssm_tensor,),
        target=target_class,
        n_steps=250
    )

    # seq_attributions => [1, seq_len, embedding_dim]
    seq_attributions_np = seq_attributions.detach().cpu().numpy()
    # Average over embedding_dim, resulting in [1, seq_len]
    seq_importances_per_sample = seq_attributions_np.mean(axis=2)
    single_seq_importance = seq_importances_per_sample[0]

    print("Sequence length: ", len(seq), " | First 50 characters of the sequence: ", seq[:50], "...")
    print("Attribution shape: ", single_seq_importance.shape)

    # Visualization: Bar chart
    visualize_attributions(single_seq_importance.reshape(1, -1),
                           title="GFP Sequence Importance (Bar)")

    # Visualization: Wavy pattern
    visualize_wavy_colored_height(single_seq_importance,
                                  title="GFP Sequence Importance (Wavy)")

    print("Sequence importance: ", single_seq_importance, "\nLength: ", len(single_seq_importance))


if __name__ == "__main__":
    main()
