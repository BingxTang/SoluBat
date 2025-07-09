# SoluBat: Integrating Mamba Model for Superior Protein Solubility Predictions

## Introduction (SoluBat)
**SoluBat** is a sophisticated hybrid model designed for precise protein solubility prediction, incorporating the **Mamba model**.

![Figure 1](https://github.com/user-attachments/assets/9478ca65-461c-4352-ae94-4ae9a794bf2f)


## Results

### Paper Results



## Features

- **Bidirectional Mamba Model**: Integrates the strengths of RNN and CNN, ensuring efficient capture and utilization of protein sequence information.
- **Multi-Head Attention Mechanism**: Enhances model expressiveness and improves prediction accuracy.
- **Automatic PSSM Generation**: No need for manual PSSM generation; the model handles it automatically.
- **High Accuracy**: Demonstrates superior performance on multiple benchmark datasets, surpassing existing state-of-the-art models.

## Requirement

Please make sure you have installed Anaconda3 or Miniconda3.

```shell
conda env create -f environment.yaml
conda activate SoluBat
```

## Configuration

All configurations for running the SoluBat model are managed through the config.ini file. Before running the model, ensure that the config.ini file is properly set up according to your data and environment. Below are the key parameters used in the configuration:

- **[General]**: The General Settings section in the config.ini file specifies the basic configurations required to run the SoluBat model. These settings are crucial for setting up the training environment and controlling key aspects of the training process.

- **[Database]**: The Database Settings section specifies paths and locations for the data that will be used for training and testing the model, as well as where the results and models will be saved.

- **[Model]**: The Model Parameters section contains configurations that define the architecture and behavior of the SoluBat model. These parameters are essential for controlling how the model processes the input data and learns from it.

## Usage

```shell
python SoluTrain.py
```

## Contributing

Contributions and suggestions from the community are welcome! If you find a bug or have an improvement suggestion, please submit an issue or a pull request.

## License

This project is licensed under the MIT License.
