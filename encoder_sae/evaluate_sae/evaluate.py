import argparse
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.append("../")
from shared.sae_actions import load_pretrained_sae
from shared.models import MiniPileDataset
from torch.utils.data import DataLoader

from tqdm import tqdm

def calculate_metrics(sae, data_loader):
    """
    Calculate MSE reconstruction loss, L1 norm, and L0 norm for the given embeddings
    using DataLoader
    
    Returns:
        tuple: (mse_loss, l1_norm, l0_norm)
    """
    device = "cuda:0"
    total_mse = 0.0
    total_l1 = 0.0
    total_l0 = 0.0

    dataset_length = len(data_loader.dataset)
    
    for _, batch_embeddings in tqdm(data_loader, desc="Calculating metrics"):
        batch = batch_embeddings.to(device)
        
        with torch.no_grad():
            # Get reconstructions and activations
            f = sae.encode(batch)
            y = sae.decode(f)

            # Calculate MSE loss
            mse = torch.sum((y - batch) ** 2).item()
            
            # Calculate L1 norm (average absolute activation values)
            l1 = torch.sum(torch.abs(f)).item()
            
            # Calculate L0 norm (average number of non-zero activations)
            l0 = torch.sum((f != 0).float()).item()
            
            # Update totals
            total_mse += mse 
            total_l1 += l1 
            total_l0 += l0 
    
    # Calculate averages
    avg_mse = total_mse / dataset_length
    avg_l1 = total_l1 / dataset_length
    avg_l0 = total_l0 / dataset_length
    
    return avg_mse, avg_l1, avg_l0

def main():
    parser = argparse.ArgumentParser(description='Evaluate SAE on test embeddings')
    parser.add_argument('--sae-dir', type=str, required=True,
                      help='Directory containing the SAE model (with config.json and sae.pkl)')
    parser.add_argument('--test-embeddings', type=str, required=True,
                      help='Path to the test embeddings .mnpypy file')
    parser.add_argument('--test-sentences', type=str, required=True,
                      help='Path to the test sentences file')
    parser.add_argument('--batch-size', type=int, default=512,
                      help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    # Load the SAE
    print(f"Loading SAE from {args.sae_dir}")
    sae = load_pretrained_sae(args.sae_dir)
    sae.eval()
    
    # Create dataset and dataloader
    print(f"Loading test data...")
    dataset = MiniPileDataset(args.test_sentences, args.test_embeddings)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Calculate metrics
    print("Calculating metrics...")
    mse, l1_norm, l0_norm = calculate_metrics(sae, data_loader)
    
    # Print results
    print("\nResults:")
    print(f"MSE Reconstruction Loss: {mse:.6f}")
    print(f"L1 Norm (mean activation): {l1_norm:.6f}")
    print(f"L0 Norm (fraction non-zero): {l0_norm:.6f}")

if __name__ == "__main__":
    main()
