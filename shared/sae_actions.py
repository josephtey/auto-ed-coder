import os
import pickle
import numpy as np
import torch
from tqdm import tqdm

from shared.sparse_autoencoder import SparseAutoencoder, SparseAutoencoderConfig
import json

def load_pretrained_sae(sae_base_path):
    """
    Load a Sparse Autoencoder model from a folder containing config.json and sae.pkl
    
    Args:
        sae_base_path: Path to folder containing config.json and sae.pkl
        
    Returns:
        Loaded SparseAutoencoder model
    """
    # Load the configuration from the JSON file
    config_path = os.path.join(sae_base_path, "config.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    # Load the pre-trained model from the pickle file
    sae_config = SparseAutoencoderConfig(
        d_model=config["dimensions"],
        d_sparse=8 * config["dimensions"],
        sparsity_alpha=config["sparsity_alpha"],
    )
    sae = SparseAutoencoder(sae_config)
    model_path = os.path.join(sae_base_path, "sae.pkl")
    with open(model_path, "rb") as f:
        model_state_dict = pickle.load(f)
        sae.load_state_dict(model_state_dict)
        
    return sae

def sae_featurize_data(data, sae, output_file=None):
    """
    Featurize data with SAEs directly.
    
    Args:
        data: Dataset containing sentences and embeddings
        sae: Trained SparseAutoencoder model
        output_file: Optional path to save feature registry
        
    Returns:
        Feature registry numpy array of shape (n_features, n_examples)
    """
    # Get dimensions
    n_examples = len(data.sentences)
    n_features = sae.encoder.weight.shape[0]  # Number of features in SAE
    
    # Create feature registry array
    if output_file:
        feature_registry = np.memmap(
            output_file,
            dtype="float32",
            mode="w+",
            shape=(n_features, n_examples)
        )
    else:
        feature_registry = np.zeros((n_features, n_examples), dtype="float32")
    
    # Choose appropriate tqdm version
    try:
        # Check if we're in a notebook environment
        get_ipython()
        from tqdm.notebook import tqdm as progress_bar
    except:
        from tqdm import tqdm as progress_bar
    
    # Process each example with progress bar
    for i, (sentence, embedding) in enumerate(progress_bar(zip(data.sentences, data.embeddings), 
                                                         total=n_examples, 
                                                         desc="Featurizing data")):
        # Convert embedding to tensor
        embedding = torch.tensor(embedding)
        
        # Get feature activations from SAE
        with torch.no_grad():
            feature_activations = sae.forward(embedding)[1]
        
        # Store in feature registry
        feature_registry[:, i] = feature_activations.detach().numpy()
        
        # Flush to disk periodically if using memmap
        if output_file and (i + 1) % 1000 == 0:
            feature_registry.flush()
    
    # Final flush if using memmap
    if output_file:
        feature_registry.flush()
        
    return feature_registry