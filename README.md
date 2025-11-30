# Plant Disease Classification Project

## ✅ Project Status: COMPLETED

**Final Test Accuracy: 90.29%** | Top-5 Accuracy: 99.55%

This project implements a deep learning-based plant disease classification system using transfer learning with **ResNet50**. The model achieved excellent performance on the PlantVillage dataset with 90.29% test accuracy.

## 🎯 Quick Results Summary
- **Dataset**: PlantVillage - 20,638 images, 16 plant disease classes
- **Test Accuracy**: 90.29%
- **Top-5 Accuracy**: 99.55%
- **Training Time**: ~15.75 hours (50 epochs, CPU-only)
- **Best Epoch**: 46
- **Validation Accuracy**: 90.09%

For detailed results, see [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md)

## Overview
This project implements a deep learning-based plant disease classification system using transfer learning with **ResNet50**. The model is trained on the PlantVillage dataset to automatically identify diseased and healthy plant leaves.

## Project Structure
```
project machine/
├── config.yaml                 # Main configuration file
├── requirements.txt            # Python dependencies
├── data/
│   ├── raw/                   # Original PlantVillage dataset
│   └── processed/             # Train/val/test splits
│       ├── train/
│       ├── val/
│       └── test/
├── src/
│   ├── preprocessing/         # Data preprocessing utilities
│   │   ├── data_loader.py    # Dataset loading and DataLoader creation
│   │   ├── augmentation.py   # Data augmentation pipelines
│   │   ├── train_val_test_split.py  # Dataset splitting
│   │   └── preprocess.py     # Main preprocessing pipeline
│   ├── models/                # Model architecture and training
│   │   ├── resnet_model.py   # ResNet50 implementation
│   │   ├── train.py          # Training loop and Trainer class
│   │   └── run_training.py   # Main training script
│   ├── evaluation/            # Evaluation metrics and testing
│   │   ├── metrics.py        # Metrics calculation
│   │   └── evaluate.py       # Model evaluation script
│   └── visualization/         # Plotting and visualization
│       └── plots.py          # Visualization utilities
├── notebooks/                 # Jupyter notebooks for exploration
├── results/                   # Training outputs
│   ├── models/               # Saved model checkpoints
│   ├── logs/                 # TensorBoard logs
│   ├── evaluation/           # Evaluation results
│   ├── figures/              # Plots and visualizations
│   └── metrics/              # Metrics JSON files
└── README.md                  # This file
```

## Features
- **Transfer Learning**: Pre-trained ResNet50 with ImageNet weights
- **Data Augmentation**: Advanced augmentations using Albumentations
- **Training Pipeline**: Complete training loop with validation, checkpointing, early stopping
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, confusion matrix, ROC-AUC
- **Visualization**: Training curves, confusion matrices, per-class metrics, sample predictions
- **Configuration Management**: YAML-based configuration for easy experimentation
- **TensorBoard Integration**: Real-time training monitoring

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended) or CPU

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd "project machine"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the PlantVillage dataset:
- Download from [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- Extract to `data/raw/PlantVillage/`

## Usage

### 1. Data Preprocessing
Prepare the dataset by creating train/val/test splits:

```bash
cd src/preprocessing
python preprocess.py --raw-dir ../../data/raw/PlantVillage --processed-dir ../../data/processed
```

### 2. Training
Train a model using the configuration file:

```bash
cd src/models
python run_training.py --config ../../config.yaml
```

Or with command-line arguments:

```bash
python run_training.py \
  --model resnet50 \
  --batch-size 32 \
  --epochs 50 \
  --lr 0.001 \
  --optimizer adam
```

### 3. Evaluation
Evaluate the trained model on test set:

```bash
cd src/evaluation
python evaluate.py \
  --checkpoint ../../results/models/best_model.pth \
  --test-dir ../../data/processed/test
```

### 4. Visualization
Create evaluation report with all visualizations:

```python
from src.visualization.plots import create_evaluation_report

create_evaluation_report(
    metrics_path='results/evaluation/test_metrics.json',
    history_path='results/models/training_history.json',
    output_dir='results/figures'
)
```

## Configuration

The `config.yaml` file contains all hyperparameters and settings:
The `config.yaml` file contains all hyperparameters and settings:

### Key Parameters
- **Model**: ResNet50 with transfer learning
- **Image Size**: 224×224 (standard for CNNs)
- **Batch Size**: 32 (adjust based on GPU memory)
- **Learning Rate**: 0.001
- **Optimizer**: adam, sgd, or adamw
- **Augmentation**: Advanced (Albumentations) or basic (torchvision)

## Model Architecture

### ResNet50
- **Architecture**: Deep residual network with 50 layers
- **Pre-training**: ImageNet weights
- **Parameters**: ~25M total, ~500K trainable (with frozen backbone)
- **Advantages**: 
  - Robust and reliable
  - Well-tested on plant disease classification
  - Good balance of accuracy and speed
  - Skip connections prevent vanishing gradients
## Training Strategy

### Transfer Learning Workflow
1. **Phase 1**: Train with frozen backbone
   - Only classification head is trained
   - Fast convergence, prevents overfitting
   
2. **Phase 2** (Optional): Fine-tune entire network
   - Unfreeze backbone layers
   - Use lower learning rate
   - Further improve accuracy
### Expected Performance
- **Accuracy**: 95-98% on PlantVillage dataset
- **Training Time**: ~2-3 hours on GPU (50 epochs)
- **Inference**: Real-time on GPU, <1 second per image on CPU dataset
- **Training Time**: 
  - ResNet50: ~2-3 hours on GPU (50 epochs)
  - EfficientNet-B0: ~1-2 hours on GPU

### Output Files
- `best_model.pth`: Best model checkpoint
- `training_history.json`: Training metrics
- `test_metrics.json`: Test set evaluation
- `confusion_matrix.png`: Confusion matrix visualization
- `training_history.png`: Loss and accuracy curves

## 📊 Results

### Model Performance
The trained ResNet50 model achieved excellent results on the test set:

| Metric | Value |
|--------|-------|
| Test Accuracy | 90.29% |
| Top-5 Accuracy | 99.55% |
| Macro F1-Score | 88.41% |
| Weighted F1-Score | 90.05% |
| Cohen's Kappa | 0.8938 |

### Best Performing Classes
- **Pepper__bell___healthy**: 98.02% F1
- **Tomato__Tomato_YellowLeaf__Curl_Virus**: 97.80% F1
- **Potato___Early_blight**: 97.69% F1

### Challenging Classes
- **Tomato_Early_blight**: 65.27% F1 (low recall)
- **Potato___healthy**: 83.64% F1 (precision issues)

### Generated Outputs
All evaluation results are saved in:
- **Metrics**: `results/evaluation/test_metrics.json`
- **Report**: `results/evaluation/classification_report.txt`
- **Visualizations**: `results/figures/`
  - Training history curves
  - Confusion matrices (raw and normalized)
  - Per-class metrics comparison

See [`RESULTS_SUMMARY.md`](RESULTS_SUMMARY.md) for complete analysis.

## Advanced Usage

### Custom Data Augmentation
Edit `src/preprocessing/augmentation.py` to customize augmentations:
```python
train_transform = get_training_transforms(
    image_size=224,
    advanced=True  # Use Albumentations
)
```

### Learning Rate Scheduling
Configure in `config.yaml`:
```yaml
training:
  scheduler: 'reduce_on_plateau'
  scheduler_params:
    factor: 0.1
    patience: 5
```

### Early Stopping
```yaml
training:
  early_stopping_patience: 10
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in config
- Use smaller model (e.g., efficientnet_b0 instead of b7)

### Slow Training
- Increase `num_workers` for data loading
- Enable `cudnn_benchmark: true`
- Use mixed precision training

### Poor Performance
- Increase `num_epochs`
- Try different learning rates
- Enable advanced augmentation
- Unfreeze backbone for fine-tuning

## Citation

If you use this project, please cite the PlantVillage dataset:
```
Hughes, D. P., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. arXiv preprint arXiv:1511.08060.
```

## License

This project is for educational purposes. The PlantVillage dataset is publicly available for research use.

## Contact

For questions or issues, please open an issue on GitHub.

## Acknowledgments

- PlantVillage dataset creators
- PyTorch and torchvision teams
- Albumentations library
- TIMM (PyTorch Image Models) library
# CSC422_Final_Aguirre_Nathan
