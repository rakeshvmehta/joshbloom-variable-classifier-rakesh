import torch
import numpy as np
from process_galaxy_dataset import get_data_loaders
from galaxy_embeddings import GalaxyEmbedder, visualize_similar_galaxies
import os
import pickle
from tqdm import tqdm

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Get data loaders
    print("Loading galaxy dataset...")
    data = get_data_loaders(
        image_dir="training_images",
        labels_file="training_classifications.csv",
        downsized_dir="downsized_galaxy_images",
        batch_size=64,  # Smaller batch size for GPU memory
        num_workers=4,
        cache_size=1000
    )
    
    # Initialize embedder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    embedder = GalaxyEmbedder(model_name='resnet50', device=device)
    
    # Create directory for embeddings
    os.makedirs('embeddings', exist_ok=True)
    
    # Compute embeddings for training set
    print("\nComputing embeddings for training set...")
    train_embeddings, train_ids = embedder.compute_embeddings(data['train_loader'])
    
    # Compute embeddings for validation set
    print("\nComputing embeddings for validation set...")
    val_embeddings, val_ids = embedder.compute_embeddings(data['val_loader'])
    
    # Save embeddings
    print("\nSaving embeddings...")
    with open('embeddings/train_embeddings.pkl', 'wb') as f:
        pickle.dump({'embeddings': train_embeddings, 'ids': train_ids}, f)
    with open('embeddings/val_embeddings.pkl', 'wb') as f:
        pickle.dump({'embeddings': val_embeddings, 'ids': val_ids}, f)
    
    # Demonstrate similar galaxy search
    print("\nDemonstrating similar galaxy search...")
    
    # Pick a random query galaxy from validation set
    query_idx = np.random.randint(len(val_embeddings))
    query_embedding = val_embeddings[query_idx]
    query_id = val_ids[query_idx]
    
    # Find similar galaxies in training set
    similar_ids, similarities = embedder.find_similar_galaxies(
        query_embedding, 
        train_embeddings,
        gallery_ids=train_ids,
        k=5,
        metric='cosine'
    )
    
    # Visualize results
    print("\nVisualizing similar galaxies...")
    visualize_similar_galaxies(
        data['full_dataset'],
        query_id,
        similar_ids,
        similarities,
        save_path='embeddings/similar_galaxies.png'
    )
    
    print("\nAnalysis complete! Results saved in embeddings/ directory")
    print("- Training embeddings shape:", train_embeddings.shape)
    print("- Validation embeddings shape:", val_embeddings.shape)
    print("- Example visualization saved as 'embeddings/similar_galaxies.png'")

if __name__ == "__main__":
    main() 