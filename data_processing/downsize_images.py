import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, morphology, measure
from skimage.color import rgb2gray
import glob
import time

def auto_crop(image, threshold=0.05):
    """
    Automatically crop empty space around an image.
    
    Args:
        image: The input image
        threshold: Pixel value to consider as foreground
        
    Returns:
        Cropped image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = rgb2gray(image)
    else:
        gray = image
        
    # Create binary image
    binary = gray > threshold
    
    # Apply morphological closing to connect nearby regions
    closed = morphology.closing(binary, morphology.square(5))
    
    # Find the bounding box of the foreground
    label_img = measure.label(closed)
    regions = measure.regionprops(label_img)
    
    if not regions:
        return image  # Return original if no regions found
    
    # Get the largest region
    largest_region = max(regions, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest_region.bbox
    
    # Add a small padding (10% of dimensions)
    height, width = image.shape[:2]
    pad_h = int(height * 0.1)
    pad_w = int(width * 0.1)
    
    minr = max(0, minr - pad_h)
    minc = max(0, minc - pad_w)
    maxr = min(height, maxr + pad_h)
    maxc = min(width, maxc + pad_w)
    
    # Crop the image
    return image[minr:maxr, minc:maxc]

def downsize_image(image_path, output_path, target_reduction=30, output_size=(128, 128)):
    """
    Downsize an image through cropping and resampling to achieve approximately
    the target reduction factor in file size.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image
        target_reduction: Target reduction factor for file size
        output_size: Tuple of (width, height) for the final image size
    
    Returns:
        Tuple of (original_image, downsized_image)
    """
    try:
        # Read the image
        img = io.imread(image_path)
        
        # Auto-crop the image
        cropped = auto_crop(img)
        
        # Resize the image to the standard output size
        resized = transform.resize(cropped, output_size, anti_aliasing=True, preserve_range=True).astype(np.uint8)
        
        # Save the downsized image
        io.imsave(output_path, resized, check_contrast=False)
        
        return img, resized
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None, None

def process_all_images(input_dir, output_dir, target_reduction=30, num_examples=4, max_images=None, output_size=(128, 128)):
    """
    Process all images in a directory and save downsized versions.
    Also creates a comparison plot for a few examples.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output images
        target_reduction: Target reduction factor for file size
        num_examples: Number of example images to plot
        max_images: Maximum number of images to process (None for all images)
        output_size: Tuple of (width, height) for all output images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
    
    # Limit the number of images if specified
    if max_images is not None:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images. Processing...")
    start_time = time.time()
    
    # Process each image
    total_orig_size = 0
    total_new_size = 0
    example_images = []
    
    for i, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, filename)
        
        # Process the image
        orig_img, downsized_img = downsize_image(img_path, output_path, target_reduction, output_size)
        
        if orig_img is not None:
            # Track file sizes
            orig_size = os.path.getsize(img_path)
            new_size = os.path.getsize(output_path)
            total_orig_size += orig_size
            total_new_size += new_size
            
            # Store example images for plotting
            if i < num_examples:
                example_images.append((filename, orig_img, downsized_img, orig_size, new_size))
        
        # Print progress every 10 images
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")
    
    # Calculate overall reduction
    if total_orig_size > 0:
        reduction_factor = total_orig_size / total_new_size
        print(f"\nOverall size reduction: {reduction_factor:.2f}x")
        print(f"Original total size: {total_orig_size / (1024*1024):.2f} MB")
        print(f"New total size: {total_new_size / (1024*1024):.2f} MB")
    
    print(f"Processing completed in {time.time() - start_time:.2f} seconds")
    
    # Plot example images
    plot_examples(example_images)

def plot_examples(example_images):
    """
    Create comparison plots for example images.
    
    Args:
        example_images: List of tuples (filename, original, downsized, orig_size, new_size)
    """
    if not example_images:
        return
    
    n = len(example_images)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4*n))
    
    for i, (filename, orig, downsized, orig_size, new_size) in enumerate(example_images):
        # Single example case
        if n == 1:
            ax1, ax2 = axes
        else:
            ax1, ax2 = axes[i]
        
        # Original image
        ax1.imshow(orig)
        ax1.set_title(f"Original: {filename}\nSize: {orig_size/1024:.1f} KB, {orig.shape[0]}x{orig.shape[1]}")
        ax1.axis('off')
        
        # Downsized image
        ax2.imshow(downsized)
        ax2.set_title(f"Downsized: {filename}\nSize: {new_size/1024:.1f} KB, {downsized.shape[0]}x{downsized.shape[1]}\nReduction: {orig_size/new_size:.1f}x")
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('image_comparison.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    input_directory = "training_images"
    output_directory = "downsized_galaxy_images"
    
    # Standard output size for all images (width, height)
    output_size = (128, 128)
    
    # Process all images with a target reduction factor of 30
    # Change max_images to limit processing (None for all images)
    process_all_images(input_directory, output_directory, target_reduction=30, max_images=3, output_size=output_size)