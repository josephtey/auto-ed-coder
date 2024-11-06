import os
import pickle
import numpy as np
import torch

from sparse_autoencoder import SparseAutoencoder, SparseAutoencoderConfig
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

def sae_featurize_data(data, sae, ref_data, output_file=None):
    """
    Featurize data with SAEs using reference embeddings.
    
    Args:
        data: DataFrame with 'label' and 'text' columns
        sae: Trained SparseAutoencoder model
        ref_data: MiniPileDataset containing sentences and embeddings
        output_file: Optional path to save feature registry
        
    Returns:
        Feature registry numpy array of shape (n_features, n_examples)
    """
    # Get dimensions
    n_examples = len(data)
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
    
    # Build lookup dictionary from sentences to embeddings
    embedding_lookup = {sent: emb for sent, emb in zip(ref_data.sentences, ref_data.embeddings)}
    
    # Process each example
    for i, row in enumerate(data.itertuples()):
        # Get embedding for this text
        text = row.text
        if text not in embedding_lookup:
            raise ValueError(f"Text not found in reference dataset: {text}")
        embedding = torch.tensor(embedding_lookup[text])
        
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
    