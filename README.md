# SoluBat: A Bidirectional Mamba Framework for High-Throughput Protein Solubility Prediction in Bioprocess Optimization

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

| Parameter   | Description                                 | 
| ----------- | ------------------------------------------- | 
| data\_root  | Root directory for raw data                 | 
| num\_folds  | Number of cross-validation splits           |
| fold\_idx   | Fold index (`-1` for all, `0-6` for single) |
| batch\_size | Training batch size                         |
| lr          | Initial learning rate                       |
| max\_epochs | Maximum training epochs                     | 


## Usage

```shell
python scripts/train.py
```

## Contributing

Contributions and suggestions from the community are welcome! If you find a bug or have an improvement suggestion, please submit an issue or a pull request.

## License

This project is licensed under the MIT License.
