# SoluBat: Integrating Mamba Model for Superior Protein Solubility Predictions

## Introduction (SoluBat)
SoluBat is a sophisticated hybrid model designed for precise protein solubility prediction, incorporating the **Mamba model**.

![Model](https://github.com/user-attachments/assets/d7819607-3f5b-49d5-99bf-adba19b3eb9b)

## Results

### Paper Results

![4c096e7a53bf774c0443041b6c281043](https://github.com/user-attachments/assets/667d45bb-63bc-4385-af0b-a77beea2ed0f)

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

```
```

## Contributing

Contributions and suggestions from the community are welcome! If you find a bug or have an improvement suggestion, please submit an issue or a pull request.

## License

This project is licensed under the MIT License.
