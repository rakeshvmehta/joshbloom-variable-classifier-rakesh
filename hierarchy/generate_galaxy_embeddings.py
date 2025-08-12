"""
Galaxy Class Embedding Generator

This script generates unitsphere embeddings for galaxy morphology classes based on the 
hierarchical relationships defined in GalaxyHierarchy. The embeddings are created such 
that the dot product between any two class embeddings corresponds to their semantic 
similarity as defined by the hierarchy.

Based on the semantic embeddings approach from:
https://github.com/cvjena/semantic-embeddings
"""

import numpy as np
import pickle
import os
from galaxy_hierarchy import GalaxyHierarchy


def unitsphere_embedding(class_sim):
    """
    Finds an embedding of `n` classes on a unit sphere in `n`-dimensional space, so that their dot products correspond
    to pre-defined similarities.
    
    Args:
        class_sim: `n-by-n` matrix specifying the desired similarity between each pair of classes.
    
    Returns: 
        `n-by-n` matrix with rows being the locations of the corresponding classes in the embedding space.
    """
    
    # Check arguments
    if (class_sim.ndim != 2) or (class_sim.shape[0] != class_sim.shape[1]):
        raise ValueError('Given class_sim has invalid shape. Expected: (n, n). Got: {}'.format(class_sim.shape))
    if (class_sim.shape[0] == 0):
        raise ValueError('Empty class_sim given.')
    
    # Place first class
    nc = class_sim.shape[0]
    embeddings = np.zeros((nc, nc))
    embeddings[0,0] = 1.
    
    # Iteratively place all remaining classes
    for c in range(1, nc):
        embeddings[c, :c] = np.linalg.solve(embeddings[:c, :c], class_sim[c, :c])
        embeddings[c, c] = np.sqrt(1. - np.sum(embeddings[c, :c] ** 2))
    
    return embeddings


def create_galaxy_embeddings(save_path='galaxy_hierarchy_embeddings.unitsphere.pickle', verbose=True):
    """
    Creates unitsphere embeddings for galaxy morphology classes based on the hierarchy.
    
    Args:
        save_path: Path where to save the embedding pickle file
        verbose: Whether to print progress information
    
    Returns:
        tuple: (embeddings, class_names, label_mappings)
    """
    
    if verbose:
        print("Loading Galaxy Hierarchy...")
    
    # Load your hierarchy
    hierarchy = GalaxyHierarchy()
    
    # Get the classification labels (37 classes)
    class_names = hierarchy.classification_labels
    n_classes = len(class_names)
    
    if verbose:
        print(f"Found {n_classes} galaxy classification classes")
        print("Computing pairwise semantic similarities...")
    
    # Create similarity matrix using your hierarchy's metric2 function
    sim_matrix = np.zeros((n_classes, n_classes))
    
    for i in range(n_classes):
        for j in range(n_classes):
            # Get the node names from the class labels
            node_i = hierarchy.label_to_node[class_names[i]]
            node_j = hierarchy.label_to_node[class_names[j]]
            
            # Compute semantic similarity using your hierarchy
            similarity = hierarchy.metric2(node_i, node_j)
            sim_matrix[i, j] = similarity
            
        if verbose and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_classes} classes")
    
    if verbose:
        print("Similarity matrix statistics:")
        print(f"  Shape: {sim_matrix.shape}")
        print(f"  Range: [{sim_matrix.min():.3f}, {sim_matrix.max():.3f}]")
        print(f"  Mean: {sim_matrix.mean():.3f}")
        print(f"  Diagonal (self-similarity): {np.diag(sim_matrix)[:5]} ...")
    
    # Generate unitsphere embeddings
    if verbose:
        print("Generating unitsphere embeddings...")
    
    try:
        embeddings = unitsphere_embedding(sim_matrix)
        
        if verbose:
            print(f"Successfully generated embeddings with shape: {embeddings.shape}")
            
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        print("This might happen if the similarity matrix is not positive semi-definite.")
        print("Trying to fix the similarity matrix...")
        
        # Make the matrix symmetric and positive semi-definite
        sim_matrix = (sim_matrix + sim_matrix.T) / 2
        eigenvals = np.linalg.eigvals(sim_matrix)
        if np.any(eigenvals < 0):
            print("Making similarity matrix positive semi-definite...")
            sim_matrix += np.eye(n_classes) * (abs(eigenvals.min()) + 1e-6)
        
        embeddings = unitsphere_embedding(sim_matrix)
        print(f"Fixed and generated embeddings with shape: {embeddings.shape}")
    
    # Create label mappings (similar to sydney-variable-classifier format)
    # ind2label: maps index to class index (for compatibility)
    # label2ind: maps class index to embedding index
    label_mappings = {
        'ind2label': list(range(n_classes)),  # [0, 1, 2, ..., n_classes-1]
        'label2ind': {i: i for i in range(n_classes)},  # {0: 0, 1: 1, ...}
        'class_names': class_names,  # The actual class name strings
        'node_names': [hierarchy.label_to_node[label] for label in class_names]  # Node names in hierarchy
    }
    
    # Save in the same format as variable classifier
    embedding_data = {
        'embedding': embeddings,
        'ind2label': label_mappings['ind2label'],
        'label2ind': label_mappings['label2ind'],
        'class_names': label_mappings['class_names'],
        'node_names': label_mappings['node_names'],
        'similarity_matrix': sim_matrix  # Save for reference
    }
    
    if verbose:
        print(f"Saving embeddings to: {save_path}")
    
    with open(save_path, 'wb') as f:
        pickle.dump(embedding_data, f)
    
    if verbose:
        print("Galaxy embeddings saved successfully!")
        print("\nEmbedding file contains:")
        print("  - 'embedding': The unitsphere embeddings matrix")
        print("  - 'ind2label': Index to label mapping")
        print("  - 'label2ind': Label to index mapping") 
        print("  - 'class_names': List of classification label strings")
        print("  - 'node_names': List of hierarchy node names")
        print("  - 'similarity_matrix': Original similarity matrix")
    
    return embeddings, class_names, label_mappings


def test_embeddings(embedding_path='galaxy_hierarchy_embeddings.unitsphere.pickle'):
    """
    Test the generated embeddings by computing some example similarities.
    
    Args:
        embedding_path: Path to the saved embedding file
    """
    
    print("Testing generated embeddings...")
    
    # Load embeddings
    with open(embedding_path, 'rb') as f:
        data = pickle.load(f)
    
    embeddings = data['embedding']
    class_names = data['class_names']
    
    print(f"Loaded embeddings shape: {embeddings.shape}")
    print(f"Number of classes: {len(class_names)}")
    
    # Test some similarities
    print("\nTesting embedding similarities (dot products):")
    
    # Self-similarity should be close to 1 (but not exactly due to numerical precision)
    self_sim = np.diag(np.dot(embeddings, embeddings.T))
    print(f"Self-similarities range: [{self_sim.min():.6f}, {self_sim.max():.6f}]")
    print(f"Self-similarities mean: {self_sim.mean():.6f}")
    
    # Show some example class similarities
    print(f"\nExample class similarities:")
    for i in range(min(5, len(class_names))):
        for j in range(i+1, min(i+4, len(class_names))):
            sim = np.dot(embeddings[i], embeddings[j])
            print(f"  '{class_names[i][:30]}...' <-> '{class_names[j][:30]}...': {sim:.4f}")
    
    print("\nEmbedding test completed!")


def main():
    """Main function to generate and test galaxy embeddings."""
    
    print("="*60)
    print("GALAXY MORPHOLOGY CLASS EMBEDDING GENERATOR")
    print("="*60)
    
    # Generate embeddings
    embeddings, class_names, mappings = create_galaxy_embeddings(verbose=True)
    
    print("\n" + "="*60)
    
    # Test the embeddings
    test_embeddings()
    
    print("\n" + "="*60)
    print("EMBEDDING GENERATION COMPLETE!")
    print("="*60)
    
    print(f"\nYou can now use the embeddings in your hierarchical CNN training!")
    print(f"Embedding file: galaxy_hierarchy_embeddings.unitsphere.pickle")
    print(f"Classes embedded: {len(class_names)}")


if __name__ == "__main__":
    main() 