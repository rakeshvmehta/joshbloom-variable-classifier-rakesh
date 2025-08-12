# Data Processing Scripts

This directory contains scripts for processing and preparing the galaxy dataset for training.

## 📁 Contents

```
data_processing/
├── process_galaxy_dataset.py    # Main dataset processing pipeline
├── downsize_images.py           # Image resizing and preprocessing
└── README.md                    # This file
```

## 🔧 Scripts

### **Process Galaxy Dataset** (`process_galaxy_dataset.py`)
- **Main purpose**: Load and prepare galaxy images and labels for training
- **Key functions**:
  - Load training images and classification labels
  - Create train/validation splits
  - Setup data loaders with proper preprocessing
  - Handle image caching for performance
- **Usage**: Imported by training scripts to get data loaders
- **Dependencies**: Requires `training_images/`, `training_classifications.csv`

### **Downsize Images** (`downsize_images.py`)
- **Main purpose**: Resize galaxy images to consistent dimensions
- **Key functions**:
  - Resize images to specified dimensions (e.g., 224x224)
  - Maintain aspect ratio or crop as needed
  - Save downsized images for faster training
- **Usage**: Run once to create `downsized_galaxy_images/` directory
- **Output**: Preprocessed images ready for training

## 🚀 Usage

### **Setting up the dataset**
```bash
# First, resize images (run once)
python downsize_images.py

# Then use in training scripts
from process_galaxy_dataset import get_data_loaders

# Get data loaders
data = get_data_loaders(
    image_dir='../training_images',
    labels_file='../training_classifications.csv',
    downsized_dir='../downsized_galaxy_images',
    batch_size=64
)

train_loader = data['train_loader']
val_loader = data['val_loader']
```

### **Data structure expected**
```
research/
├── training_images/              # Original high-res images
├── training_classifications.csv  # Labels file
├── downsized_galaxy_images/     # Preprocessed images (created by downsize_images.py)
└── data_processing/             # This directory
```

## 📊 Dataset Information

- **Original images**: High-resolution galaxy images
- **Labels**: CSV file with galaxy classifications
- **Preprocessing**: Images resized to consistent dimensions
- **Splits**: Train/validation split (configurable ratio)
- **Caching**: Optional image caching for faster loading

## 🔍 Key Functions

### **`get_data_loaders()`**
Main function that returns:
- `train_loader`: Training data loader
- `val_loader`: Validation data loader
- `class_names`: List of class names
- `num_classes`: Number of classes

### **Parameters**
- `image_dir`: Path to original images
- `labels_file`: Path to classification CSV
- `downsized_dir`: Path to preprocessed images
- `batch_size`: Batch size for training
- `num_workers`: Number of data loading workers
- `train_ratio`: Train/validation split ratio
- `cache_size`: Number of images to cache in memory

## 📝 Notes

- **Run `downsize_images.py` first** before training
- **Keep original images** for reference and potential reprocessing
- **Adjust batch_size** based on your GPU memory
- **Use caching** for faster training on repeated runs
- **Check paths** in training scripts if you move this directory

## 🔄 Workflow

1. **Prepare data**: Run `downsize_images.py` to create preprocessed images
2. **Import in training**: Use `process_galaxy_dataset.py` functions in your training scripts
3. **Train models**: Use the returned data loaders for training
4. **Evaluate**: Use validation loader for evaluation

This setup ensures consistent data preprocessing across all training approaches (hierarchical and non-hierarchical). 