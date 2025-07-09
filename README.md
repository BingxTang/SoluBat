# SoluBat: A Bidirectional Mamba Framework for High-Throughput Protein Solubility Prediction in Bioprocess Optimization

## Introduction (SoluBat)
**SoluBat** is a sophisticated hybrid model designed for precise protein solubility prediction, incorporating the **Mamba model**.

![Figure 1](https://github.com/user-attachments/assets/9478ca65-461c-4352-ae94-4ae9a794bf2f)


## Results

### Paper Results

![ee6be48ea487bb6261b1d97ae4747907](https://github.com/user-attachments/assets/cf9b75eb-4355-431d-9708-6f43f6814d4b)


## Features

- Bidirectional Mamba model boosts accuracy in protein solubility prediction.
- Dynamic gating integrates multimodal features with high efficiency.
- Near-linear complexity reduces GPU usage compared to Transformers.
- Built-in residue-level attribution enhances biological interpretability.
- Extensive benchmarking confirms strong generalization and industrial applicability.

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
