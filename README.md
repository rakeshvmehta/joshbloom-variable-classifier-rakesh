# Galaxy Classification Research Project

This repository contains comprehensive research on galaxy morphology classification using both hierarchical and non-hierarchical approaches. The project is organized for easy experimentation, hyperparameter tuning, and result comparison.

## 🏗️ Project Structure

```
research/
├── hierarchy/                    # 🎯 Hierarchical classification approach
│   ├── config.py                # Global hyperparameter configuration
│   ├── experiments/             # Experiment results and tracking
│   ├── train_hierarchical.py    # Hierarchical training script
│   ├── analyze_experiments.py   # Experiment analysis tool
│   ├── hierarchical_galaxy_cnn.py    # CNN architecture
│   ├── hierarchical_loss.py     # Loss functions
│   └── galaxy_hierarchy_embeddings.unitsphere.pickle  # Precomputed embeddings
├── non_hierarchical/            # 📊 Non-hierarchical approaches (baselines)
│   ├── scripts/                 # Training and inference scripts
│   │   ├── cnn_classifier.py       # Basic CNN classifier
│   │   ├── resnet_classifier.py    # ResNet-based classifier
│   │   ├── galaxy_embeddings.py    # Galaxy embedding generation
│   │   └── compute_galaxy_embeddings.py  # Embedding utilities
│   ├── notebooks/               # Analysis notebooks
│   │   ├── Galaxy_CNN.ipynb        # CNN training notebook
│   │   └── Lab 3.ipynb             # Lab 3 analysis notebook
│   ├── models/                  # Trained model weights
│   │   ├── best_model.pth          # Best CNN model (16MB)
│   │   └── best_resnet_model.pth   # Best ResNet model (46MB)
│   ├── plots/                   # Training visualizations
│   │   ├── resnet_training_metrics.png      # ResNet training curves
│   │   ├── training_metrics.png             # CNN training curves
│   │   ├── label_distribution_comparison.png # Dataset analysis
│   │   └── graph.jpg                        # Additional visualizations
├── data_processing/             # 🗃️ Dataset processing scripts
│   ├── process_galaxy_dataset.py # Main data pipeline
│   └── downsize_images.py       # Image preprocessing
├── training_images/              # 📸 Original galaxy images
├── downsized_galaxy_images/     # 🖼️ Preprocessed images
├── training_classifications.csv  # 🏷️ Galaxy labels
├── papers/                      # 📚 Research papers and references
└── README.md                    # 📖 This unified guide
```

## 🎯 Research Goals

### **Hierarchical Approach** (`hierarchy/`)
- **Dual-output CNN**: Generates both embeddings and classifications
- **Hierarchical loss**: Combines embedding loss with classification loss
- **Semantic relationships**: Leverages galaxy morphology hierarchies
- **Improved performance**: Better accuracy through structural understanding

### **Non-Hierarchical Approaches** (`non_hierarchical/`)
- **Standard classifiers**: CNN and ResNet with cross-entropy loss
- **Baseline comparison**: Performance reference for hierarchical improvements
- **Traditional training**: Standard deep learning approaches

## 🚀 Quick Start Guide

### **1. Setup Data Processing**
```bash
cd data_processing
python downsize_images.py  # Run once to preprocess images
```

### **2. Run Hierarchical Training** (Recommended)
```bash
cd hierarchy

# Edit hyperparameters in config.py
nano config.py

# Run training
python train_hierarchical.py

# Analyze results
python analyze_experiments.py
```

### **3. Run Non-Hierarchical Baselines**
```bash
cd non_hierarchical/scripts
python resnet_classifier.py
python cnn_classifier.py
```

## 🔧 Hyperparameter Tuning

### **Primary Configuration: `hierarchy/config.py`**

This is your **main playground** for hyperparameter tuning. All key parameters are defined here:

```python
# ===== Data Configuration =====
BATCH_SIZE = 64          # Try: 32, 64, 128
NUM_WORKERS = 4          # Adjust based on your CPU cores

# ===== Model Configuration =====
MODEL_TYPE = 'resnet18'  # Try: 'resnet18', 'resnet34', 'resnet50'
EMBEDDING_DIM = 128      # Try: 64, 128, 256

# ===== Training Configuration =====
NUM_EPOCHS = 100         # Try: 50, 100, 200
LEARNING_RATE = 0.001    # Try: 0.0001, 0.001, 0.01
OPTIMIZER = 'adam'       # Try: 'adam', 'sgd'
WEIGHT_DECAY = 1e-4      # Try: 1e-5, 1e-4, 1e-3
SCHEDULER = 'step'       # Try: 'step', 'cosine', 'plateau'

# ===== Loss Configuration =====
EMBEDDING_LOSS_TYPE = 'inv_corr'  # Try: 'inv_corr', 'squared_distance'
CLASSIFICATION_WEIGHT = 0.1        # Try: 0.05, 0.1, 0.2
EMBEDDING_WEIGHT = 1.0             # Usually keep at 1.0
```

### **Workflow for Testing Hyperparameters:**

1. **Edit `hierarchy/config.py`** - change the values you want to test
2. **Run experiment**: `python train_hierarchical.py`
3. **Check results**: Look in the new experiment folder
4. **Analyze**: `python analyze_experiments.py` to compare with previous runs
5. **Repeat**: Edit `config.py` with new values and run again

### **Example: Testing Learning Rates**
```python
# In config.py, change this line:
LEARNING_RATE = 0.0001  # Test lower learning rate

# Run experiment
python train_hierarchical.py

# Change to test higher learning rate
LEARNING_RATE = 0.01

# Run another experiment
python train_hierarchical.py

# Compare results
python analyze_experiments.py
```

## 📊 Experiment Tracking System

### **Automatic Organization**
Each experiment automatically creates:
- **Unique folder**: `experiments/exp_YYYYMMDD_HHMMSS/`
- **Configuration**: Exact settings used saved as JSON
- **Metrics**: Training curves, losses, accuracies saved as JSON
- **Models**: Checkpoints and best models saved
- **Plots**: Training curves automatically generated
- **Summary**: Master CSV file for quick comparison

### **What Gets Tracked**
- **Hyperparameters**: Learning rate, batch size, model type, etc.
- **Training metrics**: Loss curves, accuracy curves, learning rate schedules
- **Model performance**: Best validation accuracy, best validation loss
- **Training time**: Per epoch timing and total training time

### **Experiment Comparison**
```bash
cd hierarchy
python analyze_experiments.py

# Options:
# 1. View experiments summary
# 2. Plot hyperparameter comparison
# 3. Find best experiment
# 4. Analyze specific experiment
```

## 🧪 Running Experiments

### **Method 1: Modify Global Config (Recommended)**
1. **Edit `hierarchy/config.py`** - change hyperparameters
2. **Run experiment**: `python train_hierarchical.py`
3. **Results saved to**: `experiments/exp_YYYYMMDD_HHMMSS/`

### **Method 2: Command Line Overrides**
```bash
# Override specific parameters without changing config.py
python train_hierarchical.py --lr 0.0001 --batch_size 32

# Available overrides:
--lr          # Learning rate
--batch_size  # Batch size
--cls_weight  # Classification weight
--epochs      # Number of epochs
--name        # Custom experiment name
```

### **Method 3: Custom Config in Code**
```python
from train_hierarchical import run_experiment

# Run with custom config
custom_config = {
    'LEARNING_RATE': 0.0001,
    'BATCH_SIZE': 32,
    'CLASSIFICATION_WEIGHT': 0.2
}

trainer = run_experiment("custom_exp", custom_config)
```

## 📈 Analyzing Results

### **Individual Experiment Analysis**
- **Training curves**: Loss, accuracy, component losses, learning rate
- **Model checkpoints**: Best model, last model, epoch checkpoints
- **Configuration**: Exact hyperparameters used
- **Metrics**: Detailed training statistics

### **Cross-Experiment Comparison**
- **Hyperparameter impact**: See which settings work best
- **Performance ranking**: Automatically identify top experiments
- **Visual comparison**: Overlay training curves from different runs
- **Statistical analysis**: Correlation between parameters and performance

## 🔍 Key Features

### **Experiment Tracking**
- ✅ **Free & Local**: No external dependencies or costs
- ✅ **Better than W&B**: More control, faster, no internet needed
- ✅ **Easy Tweaking**: Change one file to test new settings
- ✅ **Automatic Tracking**: Everything logged automatically
- ✅ **Smart Comparison**: See which hyperparameters work best
- ✅ **Reproducible**: Exact configs saved for each experiment

### **Data Pipeline**
- **Consistent preprocessing** across all approaches
- **Image caching** for faster training
- **Train/validation splits** with configurable ratios
- **Batch processing** with adjustable sizes

## 💡 Best Practices

### **1. Hyperparameter Search Strategy**
- **Start with learning rate**: Test 0.0001, 0.001, 0.01
- **Then batch size**: Try 32, 64, 128
- **Then classification weight**: Test 0.05, 0.1, 0.2
- **Then embedding loss type**: inv_corr vs squared_distance
- **Finally optimizer/scheduler**: Adam vs SGD, different schedulers

### **2. Experiment Naming**
- Use descriptive names: `lr_0001_batch_32_cls_02`
- Include key hyperparameters in names
- Use consistent formatting

### **3. Documentation**
- Add notes to experiments in the summary CSV
- Document any data preprocessing changes
- Note hardware/software environment

## 🚨 Troubleshooting

### **Common Issues**

1. **Import errors**
   - ✅ **Fixed**: Updated import paths after reorganization
   - Ensure you're in the correct directory when running scripts

2. **Experiment fails to start**
   - Check `config.py` for valid values
   - Ensure data paths are correct
   - Verify embedding file exists

3. **Training crashes**
   - Check GPU memory (reduce batch size)
   - Verify data format matches expectations
   - Check for NaN values in loss

4. **Poor performance**
   - Try different learning rates
   - Adjust classification weight
   - Test different embedding loss types

### **Debug Mode**
```python
# Add to config.py for debugging
LOG_FREQUENCY = 1      # Log every batch
PLOT_FREQUENCY = 1     # Plot every epoch
```

## 📚 Advanced Usage

### **Custom Loss Functions**
```python
# Add new loss types to hierarchical_loss.py
# Update config.py with new options
```

### **Custom Models**
```python
# Add new model architectures to hierarchical_galaxy_cnn.py
# Update config.py MODEL_TYPE options
```

### **Automated Hyperparameter Search**
```python
# Use libraries like Optuna for advanced optimization
# Integrate with the existing experiment tracking system
```

## 🎯 Next Steps

1. **Start with baseline**: Run with current config
2. **Test learning rate**: Try 0.0001, 0.001, 0.01
3. **Test batch size**: Try 32, 64, 128
4. **Test classification weight**: Try 0.05, 0.1, 0.2
5. **Test embedding loss**: Try inv_corr vs squared_distance

## 🛠️ Requirements

- **Python**: 3.8+
- **PyTorch**: Latest stable version
- **Data Science**: NumPy, Pandas, Matplotlib, Seaborn
- **Jupyter**: For notebook analysis (optional)

## 📝 Notes

- **Data**: Keep original images in `training_images/`
- **Experiments**: All results organized in `hierarchy/experiments/`
- **Models**: Trained weights saved in respective directories
- **Configuration**: Edit `hierarchy/config.py` for hyperparameter tuning
- **Organization**: Clean separation between approaches for easy comparison

## 🔄 Complete Workflow

1. **Preprocess data** using `data_processing/` scripts
2. **Run baseline** non-hierarchical approaches for comparison
3. **Experiment with** hierarchical approaches using `config.py`
4. **Track results** automatically in organized experiment folders
5. **Compare performance** using analysis tools
6. **Iterate** on hyperparameters and architectures
7. **Document findings** in experiment summaries

This project provides a **comprehensive, professional-grade framework** for researching galaxy classification with both traditional and advanced hierarchical approaches. The unified experiment tracking system makes it easy to test different configurations and identify the best performing models.

**Happy experimenting! 🚀**
