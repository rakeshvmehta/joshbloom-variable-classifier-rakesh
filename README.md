# Galaxy Classification Research Project

This repository contains research on galaxy morphology classification using both hierarchical and non-hierarchical approaches.

## 🏗️ Project Structure

```
research/
├── hierarchy/                    # Hierarchical classification approach
│   ├── config.py                # Global hyperparameter configuration
│   ├── experiments/             # Experiment results and tracking
│   ├── train_hierarchical.py    # Hierarchical training script
│   ├── analyze_experiments.py   # Experiment analysis tool
│   └── README_EXPERIMENTS.md    # Hierarchical approach guide
├── non_hierarchical/            # Non-hierarchical approaches (baselines)
│   ├── scripts/                 # Training and inference scripts
│   ├── notebooks/               # Analysis notebooks
│   ├── models/                  # Trained model weights
│   ├── plots/                   # Training visualizations
│   └── README.md                # Non-hierarchical guide
├── data_processing/             # Dataset processing scripts
│   ├── process_galaxy_dataset.py # Main data pipeline
│   ├── downsize_images.py       # Image preprocessing
│   └── README.md                # Data processing guide
├── training_images/              # Original galaxy images
├── downsized_galaxy_images/     # Preprocessed images
├── training_classifications.csv  # Galaxy labels
├── papers/                      # Research papers and references
└── README.md                    # This file
```

## 🎯 Research Goals

### **Hierarchical Approach** (`hierarchy/`)
- Incorporate galaxy morphology hierarchies in loss functions
- Combine embedding loss with classification loss
- Improve performance by leveraging semantic relationships

### **Non-Hierarchical Approaches** (`non_hierarchical/`)
- Standard CNN and ResNet classifiers
- Baseline performance for comparison
- Traditional cross-entropy loss training

## 🚀 Quick Start

### **1. Setup Data Processing**
```bash
cd data_processing
python downsize_images.py  # Run once to preprocess images
```

### **2. Run Hierarchical Training**
```bash
cd hierarchy
# Edit config.py to set hyperparameters
python train_hierarchical.py
```

### **3. Run Non-Hierarchical Training**
```bash
cd non_hierarchical/scripts
python resnet_classifier.py
```

### **4. Analyze Results**
```bash
cd hierarchy
python analyze_experiments.py
```

## 📊 Key Features

- **Experiment Tracking**: Comprehensive logging and comparison tools
- **Hyperparameter Tuning**: Easy configuration management
- **Multiple Approaches**: Hierarchical vs non-hierarchical comparison
- **Data Pipeline**: Consistent preprocessing across all approaches
- **Visualization**: Training curves and performance analysis

## 🔬 Research Areas

### **Hierarchical Classification**
- Dual-output CNN architecture (embeddings + classifications)
- Hierarchical loss functions
- Semantic similarity in embedding space
- Galaxy morphology relationships

### **Baseline Methods**
- CNN classifiers
- ResNet transfer learning
- Standard classification approaches
- Performance benchmarking

## 📚 Documentation

- **Hierarchical**: See `hierarchy/README_EXPERIMENTS.md`
- **Non-Hierarchical**: See `non_hierarchical/README.md`
- **Data Processing**: See `data_processing/README.md`

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- NumPy, Pandas, Matplotlib
- Jupyter (for notebooks)

## 📝 Notes

- **Data**: Keep original images in `training_images/`
- **Experiments**: All results organized in `hierarchy/experiments/`
- **Models**: Trained weights saved in respective directories
- **Configuration**: Edit `hierarchy/config.py` for hyperparameter tuning

## 🔄 Workflow

1. **Preprocess data** using `data_processing/` scripts
2. **Run baseline** non-hierarchical approaches
3. **Experiment with** hierarchical approaches
4. **Compare results** using analysis tools
5. **Iterate** on hyperparameters and architectures

This project provides a comprehensive framework for researching galaxy classification with both traditional and advanced hierarchical approaches.
