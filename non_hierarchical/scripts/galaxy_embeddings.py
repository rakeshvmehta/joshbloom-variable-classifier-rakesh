import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class GalaxyEmbedder:
    def __init__(self, model_name='resnet50', embedding_dim=2048, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the Galaxy Embedder with a pre-trained model
        
        Args:
            model_name: Name of the pre-trained model to use as backbone
            embedding_dim: Dimension of the embeddings
            device: Device to run the model on
        """
        self.device = device
        self.embedding_dim = embedding_dim
        
        # Initialize the model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            # Remove the final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
        else:
            raise ValueError(f"Model {model_name} not supported yet")
        
        self.model = self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def compute_embeddings(self, dataloader: DataLoader, return_ids=True):
        """
        Compute embeddings for all images in the dataloader
        
        Args:
            dataloader: DataLoader containing the galaxy images
            return_ids: Whether to return galaxy IDs with embeddings
            
        Returns:
            embeddings: numpy array of shape (n_samples, embedding_dim)
            galaxy_ids: list of galaxy IDs if return_ids=True
        """
        all_embeddings = []
        all_ids = []
        
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            images = batch['image'].to(self.device)
            if return_ids:
                galaxy_ids = batch['galaxy_id']
                all_ids.extend(galaxy_ids)
            
            # Compute embeddings
            embeddings = self.model(images)
            embeddings = embeddings.squeeze()
            all_embeddings.append(embeddings.cpu().numpy())
        
        # Concatenate all embeddings
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        if return_ids:
            return embeddings, all_ids
        return embeddings
    
    def compute_similarity(self, emb1, emb2, metric='cosine'):
        """
        Compute similarity between two embeddings
        
        Args:
            emb1, emb2: Embeddings to compare
            metric: Similarity metric ('cosine' or 'euclidean')
            
        Returns:
            Similarity score
        """
        if metric == 'cosine':
            return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        elif metric == 'euclidean':
            return -np.linalg.norm(emb1 - emb2)
        else:
            raise ValueError(f"Metric {metric} not supported")
    
    def find_similar_galaxies(self, query_embedding, gallery_embeddings, gallery_ids=None, k=5, metric='cosine'):
        """
        Find k most similar galaxies to a query embedding
        
        Args:
            query_embedding: Embedding of the query galaxy
            gallery_embeddings: Embeddings of all galaxies to search in
            gallery_ids: IDs of the gallery galaxies (optional)
            k: Number of similar galaxies to return
            metric: Similarity metric to use
            
        Returns:
            Indices and similarities of the k most similar galaxies
            If gallery_ids provided, returns (ids, similarities)
            Otherwise returns (indices, similarities)
        """
        similarities = np.array([
            self.compute_similarity(query_embedding, gallery_emb, metric)
            for gallery_emb in gallery_embeddings
        ])
        
        # Get top k indices
        top_k_idx = np.argsort(similarities)[-k:][::-1]
        top_k_similarities = similarities[top_k_idx]
        
        if gallery_ids is not None:
            top_k_ids = [gallery_ids[i] for i in top_k_idx]
            return top_k_ids, top_k_similarities
        
        return top_k_idx, top_k_similarities

def visualize_similar_galaxies(dataset, query_id, similar_ids, similarities, save_path=None):
    """
    Visualize a query galaxy and its similar matches
    
    Args:
        dataset: GalaxyDataset instance
        query_id: ID of the query galaxy
        similar_ids: IDs of similar galaxies
        similarities: Similarity scores
        save_path: Path to save the visualization (optional)
    """
    import matplotlib.pyplot as plt
    
    n_images = len(similar_ids) + 1
    fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 4))
    
    # Plot query image
    query_idx = dataset.galaxy_ids.index(query_id)
    query_data = dataset[query_idx]
    query_image = query_data['image']
    if isinstance(query_image, torch.Tensor):
        query_image = query_image.numpy()
    if query_image.shape[0] in [1, 3]:  # CHW format
        query_image = np.transpose(query_image, (1, 2, 0))
    
    axes[0].imshow(query_image)
    axes[0].set_title(f'Query Galaxy\nID: {query_id}')
    axes[0].axis('off')
    
    # Plot similar images
    for i, (galaxy_id, similarity) in enumerate(zip(similar_ids, similarities), 1):
        idx = dataset.galaxy_ids.index(galaxy_id)
        data = dataset[idx]
        image = data['image']
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if image.shape[0] in [1, 3]:  # CHW format
            image = np.transpose(image, (1, 2, 0))
        
        axes[i].imshow(image)
        axes[i].set_title(f'Similar Galaxy {i}\nID: {galaxy_id}\nSimilarity: {similarity:.3f}')
        axes[i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show() 