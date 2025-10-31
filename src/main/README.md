# FedMVD: Dual-Granularity Federated Learning for Multi-View Bearing Fault Diagnosis

Official implementation of **FedMVD**, a federated learning framework for privacy-preserving collaborative fault diagnosis from heterogeneous multi-view sensor data.

## Overview

FedMVD addresses the challenges of statistical heterogeneity in federated learning for rotating machinery fault diagnosis by combining:
- **Client-side**: Dual-granularity hierarchical attention for multi-view feature fusion
- **Server-side**: Discrepancy-driven alignment using class-wise higher-order central moment discrepancy

## Requirements

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- Python >= 3.8
- PyTorch >= 1.10
- NumPy, SciPy
- scikit-learn

## Dataset Preparation

1. Download the rotating machinery datasets (e.g., BJTU-RAO, PU, etc.)
2. Place datasets in the `data/` directory
3. Organize data structure as follows:

<!-- ```
data/
├── CWRU/
│   ├── train/
│   └── test/
└── PU/
    ├── train/
    └── test/
``` -->

## Quick Start

### Training and Evaluation

Train FedMVD on a specific dataset:

```bash
python main.py --dataset PU --num_clients 10 --rounds 100 --local_epochs 5
```
<!-- 
**Key arguments:**
- `--dataset`: Dataset name (BJTU-RAO, PU, etc.)
- `--num_clients`: Number of federated clients
- `--rounds`: Number of communication rounds
- `--local_epochs`: Local training epochs per round 
- `--heterogeneity`: Data heterogeneity level (low, medium, high) -->


<!-- ## Configuration -->

<!-- Modify hyperparameters in `config/config.yaml`: -->

<!-- ```yaml
model:
  attention_heads: 8
  hidden_dim: 256
  
training:
  learning_rate: 0.001
  batch_size: 32
  
federated:
  aggregation: "discrepancy_driven"
  moment_order: 4
``` -->

<!-- ## Project Structure

```
FedMVD/
├── config/              # Configuration files
├── data/                # Dataset directory
├── models/              # Model implementations
│   ├── attention.py     # Dual-granularity attention module
│   ├── fedmvd.py        # Main FedMVD model
│   └── server.py        # Server-side alignment
├── utils/               # Utility functions
│   ├── data_loader.py   # Data preprocessing
│   └── metrics.py       # Evaluation metrics
├── main.py              # Training script
└── requirements.txt     # Dependencies
``` -->

<!-- 
## Citation

If you find this work helpful, please cite:

```bibtex
@article{fedmvd2024,
  title={FedMVD: Dual-Granularity Federated Learning for Multi-View Bearing Fault Diagnosis},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
``` -->



