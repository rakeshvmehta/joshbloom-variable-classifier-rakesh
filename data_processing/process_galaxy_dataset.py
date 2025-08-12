import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from skimage import io
import time
from tqdm import tqdm
from downsize_images import downsize_image

class GalaxyDataset(Dataset):
    def __init__(self, image_dir, labels_file, transform=None, downsized_dir=None, cache_size=1000):
        """
        Galaxy dataset loader with caching for faster access
        
        Args:
            image_dir: Directory with original images
            labels_file: Path to CSV file with labels
            transform: Optional transforms to apply
            downsized_dir: Directory with pre-downsized images (if available)
            cache_size: Number of images to keep in memory cache (0 to disable)
        """
        self.image_dir = image_dir
        self.transform = transform
        self.downsized_dir = downsized_dir
        self.cache_size = cache_size
        
        # Read labels
        self.labels_df = pd.read_csv(labels_file)
        
        # Clean up column names (remove newlines if present)
        self.labels_df.columns = [col.strip() for col in self.labels_df.columns]
        
        # Get list of image IDs
        self.galaxy_ids = list(self.labels_df['GalaxyID'])
        
        # Pre-compute the image paths
        self.image_paths = {}
        for galaxy_id in self.galaxy_ids:
            if self.downsized_dir is not None and os.path.exists(os.path.join(self.downsized_dir, f"{galaxy_id}.jpg")):
                # Use downsized image if available
                self.image_paths[galaxy_id] = os.path.join(self.downsized_dir, f"{galaxy_id}.jpg")
            else:
                # Use original image
                self.image_paths[galaxy_id] = os.path.join(self.image_dir, f"{galaxy_id}.jpg")
        
        # Image cache to speed up repeated access
        self.image_cache = {}
        
        # Precompute labels as a dictionary for faster lookup
        self.labels_dict = {}
        for galaxy_id in self.galaxy_ids:
            self.labels_dict[galaxy_id] = self.labels_df.loc[self.labels_df['GalaxyID'] == galaxy_id].iloc[0, 1:].values.astype(np.float32)
            
        # Prefetch some images into memory if cache is enabled
        if self.cache_size > 0:
            print(f"Prefetching {min(self.cache_size, len(self.galaxy_ids))} images into memory cache...")
            for i, galaxy_id in enumerate(self.galaxy_ids[:self.cache_size]):
                self._load_and_cache_image(galaxy_id)
                if (i+1) % 1000 == 0:
                    print(f"Prefetched {i+1} images")
    
    def _load_and_cache_image(self, galaxy_id):
        """Load an image and store it in the cache"""
        if galaxy_id in self.image_cache:
            return self.image_cache[galaxy_id]
        
        # Load the image
        image = io.imread(self.image_paths[galaxy_id])
        
        # Preprocess image
        if len(image.shape) == 2:  # Grayscale image
            image = np.expand_dims(image, axis=2)  # Add channel dimension
            
        # Ensure image is in float format and normalized
        image = image.astype(np.float32) / 255.0
        
        # Convert from HWC to CHW format
        image = np.transpose(image, (2, 0, 1))
        
        # Cache the processed image if cache is enabled
        if self.cache_size > 0:
            # If cache is full, remove a random item
            if len(self.image_cache) >= self.cache_size:
                remove_key = list(self.image_cache.keys())[0]
                del self.image_cache[remove_key]
            
            # Add to cache
            self.image_cache[galaxy_id] = image
            
        return image
    
    def __len__(self):
        return len(self.galaxy_ids)
    
    def __getitem__(self, idx):
        galaxy_id = self.galaxy_ids[idx]
        
        # Get image (from cache if available)
        if galaxy_id in self.image_cache:
            image = self.image_cache[galaxy_id]
        else:
            image = self._load_and_cache_image(galaxy_id)
        
        # Get labels from precomputed dictionary
        labels = self.labels_dict[galaxy_id]
        
        # Apply transforms if specified
        if self.transform:
            # Convert back to HWC format for transforms
            image_hwc = np.transpose(image, (1, 2, 0)) 
            transformed = self.transform(image_hwc)
            # Convert back if transform didn't already do it
            if not isinstance(transformed, torch.Tensor):
                transformed = torch.from_numpy(
                    np.transpose(transformed, (2, 0, 1))
                ).float()
            image = transformed
        else:
            # Convert to PyTorch tensor
            image = torch.from_numpy(image).float()
        
        return {'image': image, 'labels': torch.tensor(labels, dtype=torch.float32), 'galaxy_id': galaxy_id}

def preprocess_images(image_dir, output_dir, output_size=(128, 128), batch_size=200):
    """
    Preprocess and downsize all images in the given directory using the downsize_image function
    
    Args:
        image_dir: Directory containing original images
        output_dir: Directory to save downsized images
        output_size: Target size for all images
        batch_size: Number of images to process in each batch
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    # Process in batches
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(total_batches), desc="Preprocessing images"):
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(image_files))
        batch_files = image_files[batch_start:batch_end]
        
        for img_file in batch_files:
            img_path = os.path.join(image_dir, img_file)
            output_path = os.path.join(output_dir, img_file)
            
            # Skip if already processed
            if os.path.exists(output_path):
                continue
            
            try:
                # Use the downsize_image function from downsize_images.py
                downsize_image(img_path, output_path, target_reduction=30, output_size=output_size)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

def create_train_val_split(dataset, train_ratio=0.8, seed=42):
    """
    Split a dataset into training and validation sets
    
    Args:
        dataset: The full dataset
        train_ratio: Ratio of data for training (0.8 = 80% training, 20% validation)
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset, val_dataset
    """
    # Calculate lengths
    total_len = len(dataset)
    train_len = int(total_len * train_ratio)
    val_len = total_len - train_len
    
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    
    return train_dataset, val_dataset

def compare_distributions(train_dataset, val_dataset, dataset, num_classes=37, figsize=(15, 5)):
    """
    Compare label distributions between training and validation sets
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        dataset: Full dataset with galaxy_ids and labels_df
        num_classes: Number of label classes
        figsize: Figure size for the plot
    """
    # Get indices from train and val datasets
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    
    # Create label matrices
    train_labels = np.zeros((len(train_indices), num_classes))
    val_labels = np.zeros((len(val_indices), num_classes))
    
    # Get labels for each set
    for i, idx in enumerate(train_indices):
        galaxy_id = dataset.galaxy_ids[idx]
        train_labels[i] = dataset.labels_df.loc[dataset.labels_df['GalaxyID'] == galaxy_id].iloc[0, 1:1+num_classes].values
    
    for i, idx in enumerate(val_indices):
        galaxy_id = dataset.galaxy_ids[idx]
        val_labels[i] = dataset.labels_df.loc[dataset.labels_df['GalaxyID'] == galaxy_id].iloc[0, 1:1+num_classes].values
    
    # Calculate mean distribution for each set
    train_dist = np.mean(train_labels, axis=0)
    val_dist = np.mean(val_labels, axis=0)
    
    # Plot distributions
    plt.figure(figsize=figsize)
    
    x = np.arange(num_classes)
    width = 0.35
    
    plt.bar(x - width/2, train_dist, width, label='Training')
    plt.bar(x + width/2, val_dist, width, label='Validation')
    
    plt.xlabel('Class Index')
    plt.ylabel('Mean Probability')
    plt.title('Label Distribution Comparison: Training vs Validation')
    plt.legend()
    plt.xticks(x, [f'C{i+1}' for i in range(num_classes)], rotation=90)
    plt.tight_layout()
    plt.savefig('label_distribution_comparison.png')
    
    return train_dist, val_dist

def create_batch_loaders(train_dataset, val_dataset, batch_size=128, num_workers=4, 
                       pin_memory=True, prefetch_factor=2, persistent_workers=True):
    """
    Create DataLoader objects for training and validation sets
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size for loading
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory in CUDA
        prefetch_factor: Number of batches to prefetch per worker
        persistent_workers: Whether to keep worker processes alive between batches
    
    Returns:
        train_loader, val_loader
    """
    # Determine if prefetch_factor and persistent_workers should be used
    prefetch_arg = prefetch_factor if num_workers > 0 else None
    persistent_arg = persistent_workers if num_workers > 0 else False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_arg,
        persistent_workers=persistent_arg
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_arg,
        persistent_workers=persistent_arg
    )
    
    return train_loader, val_loader

def benchmark_loading_speed(data_loader, num_batches=20, warmup_batches=5):
    """
    Benchmark the data loading speed
    
    Args:
        data_loader: DataLoader to benchmark
        num_batches: Number of batches to time
        warmup_batches: Number of batches to skip for warmup
    """
    print("Warming up data loader...")
    # Warmup phase (don't measure this)
    for i, _ in enumerate(data_loader):
        if i >= warmup_batches - 1:
            break
    
    print(f"Starting benchmark over {num_batches} batches...")
    start_time = time.time()
    
    total_images = 0
    batch_times = []
    
    for i, batch in enumerate(data_loader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        images = batch['image']
        batch_end = time.time()
        
        batch_time = batch_end - batch_start
        batch_size = images.shape[0]
        
        total_images += batch_size
        batch_times.append((batch_time, batch_size))
        
        # Print progress for long benchmarks
        if (i + 1) % 5 == 0:
            print(f"Processed {i + 1}/{num_batches} batches")
    
    end_time = time.time()
    total_elapsed = end_time - start_time
    
    # Overall statistics
    print(f"\nOverall results:")
    print(f"Loaded {total_images} images in {total_elapsed:.2f} seconds")
    print(f"Average loading time: {total_elapsed/total_images:.6f} seconds per image")
    print(f"Images per second: {total_images/total_elapsed:.1f}")
    
    # Per-batch statistics
    batch_loading_times = [t/s for t, s in batch_times]
    avg_batch_time = sum(batch_loading_times) / len(batch_loading_times)
    print(f"\nAverage batch loading time: {avg_batch_time:.6f} seconds per image")
    
    # Find best and worst times
    best_time = min(batch_loading_times)
    worst_time = max(batch_loading_times)
    print(f"Best loading time: {best_time:.6f} seconds per image")
    print(f"Worst loading time: {worst_time:.6f} seconds per image")
    
    # If the target wasn't met, provide suggestions
    if avg_batch_time > 0.01:
        print("\nTo further improve loading times:")
        print("1. Increase cache_size parameter to cache more images")
        print("2. Try different num_workers values to find the optimal setting")
        print("3. Ensure all images are pre-processed and downsized")
        print("4. Consider using a smaller output image size")
        print("5. Use an SSD instead of HDD if possible")

def get_data_loaders(image_dir="training_images", 
                    labels_file="training_classifications.csv", 
                    downsized_dir="downsized_galaxy_images",
                    batch_size=128,
                    num_workers=4,
                    train_ratio=0.8,
                    seed=42,
                    transform=None,
                    cache_size=1000,
                    pin_memory=True,
                    prefetch_factor=2):
    """
    Main function to get data loaders for training and validation
    
    Args:
        image_dir: Directory with original images
        labels_file: Path to CSV file with labels
        downsized_dir: Directory with downsized images
        batch_size: Batch size for the data loaders
        num_workers: Number of worker processes for data loaders
        train_ratio: Ratio of data for training
        seed: Random seed for reproducibility
        transform: Optional transforms to apply to images
        cache_size: Number of images to keep in memory cache (0 to disable)
        pin_memory: Whether to pin memory in CUDA (speeds up GPU transfer)
        prefetch_factor: Number of batches to prefetch per worker
        
    Returns:
        Dictionary containing train_loader, val_loader, train_dataset, val_dataset, and full_dataset
    """
    # Create dataset
    dataset = GalaxyDataset(
        image_dir=image_dir,
        labels_file=labels_file,
        downsized_dir=downsized_dir if os.path.exists(downsized_dir) else None,
        transform=transform,
        cache_size=cache_size
    )
    
    # Create train/val split
    train_dataset, val_dataset = create_train_val_split(
        dataset, 
        train_ratio=train_ratio,
        seed=seed
    )
    
    # Create data loaders with optimized parameters
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'full_dataset': dataset
    }

if __name__ == "__main__":
    # Define paths
    image_dir = "training_images"
    labels_file = "training_classifications.csv"
    downsized_dir = "downsized_galaxy_images"
    
    # Ask user whether to preprocess images
    should_preprocess = input("\nDo you want to preprocess and downsize the images? (yes/no, default: no): ").lower().strip() == 'yes'
    
    if should_preprocess:
        # Preprocess images using downsize_image function
        print("\nPreprocessing images using downsize_image function from downsize_images.py...")
        output_size = (128, 128)  # Standard output size for all images
        preprocess_images(image_dir, downsized_dir, output_size=output_size)
    
    # Ask user about multiprocessing
    use_multiprocessing = input("\nUse multiprocessing for data loading? (yes/no, default: yes): ").lower().strip() != 'no'
    num_workers = 4 if use_multiprocessing else 0
    
    # Ask about image caching
    use_caching = input("\nUse image caching for faster loading? (yes/no, default: yes): ").lower().strip() != 'no'
    cache_size = 5000 if use_caching else 0  # Cache more images for better performance
    
    print(f"\nUsing {num_workers} worker processes for data loading")
    print(f"Image caching: {'Enabled' if use_caching else 'Disabled'} (cache size: {cache_size})")
    
    # Get data loaders and datasets with optimized settings
    data = get_data_loaders(
        image_dir=image_dir,
        labels_file=labels_file,
        downsized_dir=downsized_dir,
        batch_size=128,
        num_workers=num_workers,
        cache_size=cache_size,
        prefetch_factor=3,        # Prefetch more batches
        pin_memory=True           # Faster GPU transfer
    )
    
    # Unpack data
    train_loader = data['train_loader']
    val_loader = data['val_loader']
    train_dataset = data['train_dataset']
    val_dataset = data['val_dataset']
    dataset = data['full_dataset']
    
    print(f"Total dataset size: {len(dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    
    # Compare distributions
    print("\nComparing label distributions between training and validation sets...")
    train_dist, val_dist = compare_distributions(
        train_dataset, 
        val_dataset, 
        dataset
    )
    
    # Benchmark loading speed
    print("\nBenchmarking data loading speed...")
    benchmark_loading_speed(train_loader)
    
    print("\nDataset processing complete!")