# Experiment Tracking System for Hierarchical Galaxy Classification

This directory now contains a comprehensive experiment tracking system that replaces Weights & Biases with a local, powerful alternative.

## 🏗️ Directory Structure

```
hierarchy/
├── config.py                    # Global configuration file
├── experiments/                 # All experiment results
│   ├── exp_001_20241201_143022/
│   │   ├── config.json          # Experiment-specific config
│   │   ├── metrics.json         # Training metrics per epoch
│   │   ├── model_checkpoints/   # Model weights
│   │   ├── plots/               # Training curves
│   │   └── logs/                # Console output
│   └── experiments_summary.csv  # Master tracking file
├── train_hierarchical.py        # Enhanced training script
├── analyze_experiments.py       # Analysis and comparison tool
└── README_EXPERIMENTS.md        # This file
```

## 🚀 Quick Start

### 1. Modify Global Configuration

Edit `config.py` to set your hyperparameters:

```python
# Example: Change learning rate
LEARNING_RATE = 0.0001

# Example: Change batch size
BATCH_SIZE = 32

# Example: Change classification weight
CLASSIFICATION_WEIGHT = 0.2
```

### 2. Run an Experiment

```bash
# Run with current config
python train_hierarchical.py

# Run with custom name
python train_hierarchical.py --name "my_experiment"

# Run with custom overrides
python train_hierarchical.py --name "lr_test" --lr 0.0001 --batch_size 32
```

### 3. Analyze Results

```bash
python analyze_experiments.py
```

## 📊 What Gets Tracked

### Hyperparameters
- Learning rate, batch size, epochs
- Model architecture (ResNet type, embedding dimensions)
- Loss weights (classification_weight, embedding_weight)
- Optimizer settings (Adam/SGD, weight decay)
- Scheduler settings (step, cosine, plateau)
- Data augmentation parameters

### Training Metrics
- Loss curves (total, embedding, classification)
- Accuracy curves (training, validation)
- Learning rate schedules
- Training time per epoch

### Model Performance
- Best validation accuracy
- Best validation loss
- Final test metrics
- Model size and parameters

## 🔧 Configuration Options

### Data Configuration
```python
IMAGE_DIR = '../training_images'
LABELS_FILE = '../training_classifications.csv'
DOWNSIZED_DIR = '../downsized_galaxy_images'
BATCH_SIZE = 64
NUM_WORKERS = 4
TRAIN_RATIO = 0.8
```

### Model Configuration
```python
MODEL_TYPE = 'resnet18'  # 'resnet18', 'resnet34', 'resnet50', 'custom'
EMBEDDING_DIM = 128
```

### Training Configuration
```python
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
OPTIMIZER = 'adam'  # 'adam', 'sgd'
WEIGHT_DECAY = 1e-4
SCHEDULER = 'step'  # 'step', 'cosine', 'plateau'
```

### Loss Configuration
```python
EMBEDDING_LOSS_TYPE = 'inv_corr'  # 'inv_corr', 'squared_distance'
CLASSIFICATION_WEIGHT = 0.1
EMBEDDING_WEIGHT = 1.0
```

## 🧪 Running Experiments

### Method 1: Modify Global Config (Recommended)
1. **Edit `config.py`** - change the hyperparameters you want to test
2. **Run experiment**: `python train_hierarchical.py`
3. **Results saved to**: `experiments/exp_YYYYMMDD_HHMMSS/`

### Method 2: Command Line Overrides
```bash
# Override specific parameters without changing config.py
python train_hierarchical.py --name "lr_test" --lr 0.0001 --batch_size 32
```

### Method 3: Custom Config Overrides in Code
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

### 1. View Experiments Summary
```bash
python analyze_experiments.py
# Choose option 1: View experiments summary
```

### 2. Compare Hyperparameters
```bash
python analyze_experiments.py
# Choose option 2: Plot hyperparameter comparison
```

### 3. Find Best Experiment
```bash
python analyze_experiments.py
# Choose option 3: Find best experiment
```

### 4. Analyze Specific Experiment
```bash
python analyze_experiments.py
# Choose option 4: Analyze specific experiment
```

## 🔍 Experiment Comparison

The system automatically creates:
- **Individual experiment folders** with all results
- **Master CSV summary** (`experiments_summary.csv`) for quick comparison
- **Hyperparameter comparison plots** showing relationships between settings and performance
- **Training curve overlays** for detailed analysis

## 💡 Best Practices

### 1. Workflow for Testing Hyperparameters
1. **Edit `config.py`** with new values
2. **Run experiment**: `python train_hierarchical.py`
3. **Analyze results**: `python analyze_experiments.py`
4. **Compare with previous runs** to see what works best
5. **Repeat** with new values based on insights

### 2. Naming Conventions
- Use descriptive names: `lr_0001_batch_32_cls_02`
- Include key hyperparameters in names
- Use consistent formatting

### 3. Hyperparameter Search Strategy
- **Start with learning rate**: Test 0.0001, 0.001, 0.01
- **Then batch size**: Try 32, 64, 128
- **Then classification weight**: Test 0.05, 0.1, 0.2
- **Then embedding loss type**: inv_corr vs squared_distance
- **Finally optimizer/scheduler**: Adam vs SGD, different schedulers

### 4. Documentation
- Add notes to experiments in the summary CSV
- Document any data preprocessing changes
- Note hardware/software environment

## 🚨 Troubleshooting

### Common Issues

1. **Experiment fails to start**
   - Check `config.py` for valid values
   - Ensure data paths are correct
   - Verify embedding file exists

2. **Training crashes**
   - Check GPU memory (reduce batch size)
   - Verify data format matches expectations
   - Check for NaN values in loss

3. **Poor performance**
   - Try different learning rates
   - Adjust classification weight
   - Test different embedding loss types

### Debug Mode
```python
# Add to config.py for debugging
LOG_FREQUENCY = 1  # Log every batch
PLOT_FREQUENCY = 1  # Plot every epoch
```

## 📚 Advanced Usage

### Custom Loss Functions
```python
# Add new loss types to hierarchical_loss.py
# Update config.py with new options
```

### Custom Models
```python
# Add new model architectures to hierarchical_galaxy_cnn.py
# Update config.py MODEL_TYPE options
```

## 🎯 Where to Tweak Parameters

### **Primary Configuration File: `config.py`**

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

### **Workflow for Hyperparameter Tuning:**

1. **Edit `config.py`** - change the values you want to test
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

## 🎯 Next Steps

1. **Start with baseline**: Run with current config
2. **Test learning rate**: Try 0.0001, 0.001, 0.01
3. **Test batch size**: Try 32, 64, 128
4. **Test classification weight**: Try 0.05, 0.1, 0.2
5. **Test embedding loss**: Try inv_corr vs squared_distance

## 📞 Support

This system is designed to be self-contained and easy to use. If you encounter issues:

1. Check the error messages in the experiment logs
2. Verify configuration values are valid
3. Ensure all dependencies are installed
4. Check data paths and file formats

**Remember**: The main way to tweak parameters is by editing `config.py` and then running `python train_hierarchical.py`!

Happy experimenting! 🚀 