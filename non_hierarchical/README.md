# Non-Hierarchical Galaxy Classification

This directory contains all the non-hierarchical approaches to galaxy classification, organized for easy access and comparison.

## 📁 Directory Structure

```
non_hierarchical/
├── scripts/                     # Python scripts for training and inference
│   ├── cnn_classifier.py       # Basic CNN classifier
│   ├── resnet_classifier.py    # ResNet-based classifier
│   ├── galaxy_embeddings.py    # Galaxy embedding generation
│   └── compute_galaxy_embeddings.py  # Embedding computation utilities
├── notebooks/                   # Jupyter notebooks for analysis
│   ├── Galaxy_CNN.ipynb        # CNN training and analysis notebook
│   └── Lab 3.ipynb             # Lab 3 analysis notebook
├── models/                      # Trained model weights
│   ├── best_model.pth          # Best CNN model
│   └── best_resnet_model.pth   # Best ResNet model
├── plots/                       # Training plots and visualizations
│   ├── resnet_training_metrics.png      # ResNet training curves
│   ├── training_metrics.png             # CNN training curves
│   ├── label_distribution_comparison.png # Dataset analysis
│   └── graph.jpg                        # Additional visualizations
└── results/                     # Additional results and outputs
```

## 🔧 Scripts

### **CNN Classifier** (`scripts/cnn_classifier.py`)
- Basic convolutional neural network for galaxy classification
- Simple architecture for baseline comparison
- Training and evaluation functions

### **ResNet Classifier** (`scripts/resnet_classifier.py`)
- ResNet-based classifier for improved performance
- Transfer learning from pre-trained models
- Advanced training techniques

### **Galaxy Embeddings** (`scripts/galaxy_embeddings.py`)
- Generate embeddings for galaxy images
- Support for different embedding strategies
- Embedding analysis and visualization

### **Compute Embeddings** (`scripts/compute_galaxy_embeddings.py`)
- Utility functions for embedding computation
- Batch processing capabilities
- Embedding storage and retrieval

## 📊 Notebooks

### **Galaxy CNN** (`notebooks/Galaxy_CNN.ipynb`)
- Complete CNN training pipeline
- Data loading and preprocessing
- Training visualization and analysis

### **Lab 3** (`notebooks/Lab 3.ipynb`)
- Lab 3 analysis and experiments
- Additional galaxy classification approaches
- Comparative analysis

## 🎯 Models

### **Best CNN Model** (`models/best_model.pth`)
- Trained CNN classifier weights
- Baseline performance reference
- 16MB model file

### **Best ResNet Model** (`models/best_resnet_model.pth`)
- Trained ResNet classifier weights
- Improved performance reference
- 46MB model file

## 📈 Plots

### **Training Metrics**
- `resnet_training_metrics.png`: ResNet training curves
- `training_metrics.png`: CNN training curves
- `label_distribution_comparison.png`: Dataset analysis
- `graph.jpg`: Additional visualizations

## 🚀 Usage

### **Running Scripts**
```bash
# Run CNN classifier
cd scripts
python cnn_classifier.py

# Run ResNet classifier
python resnet_classifier.py

# Generate embeddings
python galaxy_embeddings.py
```

### **Opening Notebooks**
```bash
# Start Jupyter
jupyter notebook notebooks/

# Or open specific notebook
jupyter notebook notebooks/Galaxy_CNN.ipynb
```

### **Loading Models**
```python
import torch

# Load CNN model
cnn_model = torch.load('models/best_model.pth')

# Load ResNet model
resnet_model = torch.load('models/best_resnet_model.pth')
```

## 🔍 Comparison with Hierarchical Approach

This directory contains the **non-hierarchical** approaches that:
- Use standard classification loss (cross-entropy)
- Don't incorporate hierarchical relationships
- Serve as baseline comparisons

Compare these results with the **hierarchical** approach in the `hierarchy/` directory to see the benefits of incorporating galaxy morphology hierarchies.

## 📝 Notes

- All scripts are self-contained and can be run independently
- Models are trained on the same dataset for fair comparison
- Plots show training progress and final performance
- Use these as baselines when evaluating hierarchical improvements 