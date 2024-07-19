import torch.optim as optim
import torch

import sys

sys.path.append("../")
from shared.sparse_autoencoder import SparseAutoencoder, SparseAutoencoderConfig
from shared.models import MiniPileDataset

from datetime import datetime
import pickle
import json
import os
import argparse
from tqdm import tqdm


def main(args):
    with open(args.dataset_filename, "rb") as f:
        loaded_dataset = pickle.load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_subfolder = os.path.join(args.run_folder, timestamp)
    os.makedirs(run_subfolder, exist_ok=True)

    # Config
    config = {
        "batch_size": args.batch_size,
        "dimensions": args.dimensions,
        "sparsity_alpha": args.sparsity_alpha,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "sparsity_scale": args.sparsity_scale,
    }

    # Assuming `dataset` is a PyTorch Dataset loaded and ready to use
    data_loader = torch.utils.data.DataLoader(
        loaded_dataset, batch_size=config["batch_size"], shuffle=True
    )

    # Initialize the model
    sae_config = SparseAutoencoderConfig(
        d_model=config["dimensions"],
        d_sparse=8 * config["dimensions"],
        sparsity_alpha=config["sparsity_alpha"],
    )
    model = SparseAutoencoder(sae_config)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Training loop
    with open(f"{run_subfolder}/training_output.out", "w") as out_file:
        num_epochs = config["num_epochs"]
        for epoch in range(num_epochs):
            total_loss = 0

            for sentences, embeddings in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()

                # Assuming data is already on the correct device and in the correct format
                _, _, loss, _ = model.forward(
                    embeddings,
                    return_loss=True,
                    sparsity_scale=config["sparsity_scale"],
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                # Print the loss for every batch
                batch_loss_str = f"Batch Loss: {loss.item()}"
                out_file.write(batch_loss_str + "\n")

            epoch_loss_str = (
                f"Epoch {epoch+1}, Average Loss: {total_loss / len(data_loader)}"
            )
            out_file.write(epoch_loss_str + "\n")
            print(epoch_loss_str)

    # Save the model to a pickle file
    model_path = f"{run_subfolder}/sae.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model.state_dict(), f)

    print(f"SAE saved to {model_path}")

    # Load the model from the pickle file
    with open(model_path, "rb") as f:
        model_state_dict = pickle.load(f)
        model.load_state_dict(model_state_dict)

    print(f"SAE loaded from {model_path}")

    # Save configuration
    config_path = f"{run_subfolder}/config.json"
    with open(config_path, "w") as config_file:
        json.dump(config, config_file)

    print(f"Configuration saved to {config_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder")
    parser.add_argument(
        "--dataset_filename", type=str, required=True, help="Path to the dataset file"
    )
    parser.add_argument(
        "--run_folder", type=str, required=True, help="Folder to save the run outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="Batch size for training"
    )
    parser.add_argument(
        "--dimensions", type=int, default=768, help="Dimensions of the input data"
    )
    parser.add_argument(
        "--sparsity_alpha", type=float, default=1, help="Sparsity alpha parameter"
    )
    parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate")
    parser.add_argument(
        "--num_epochs", type=int, default=1, help="Number of epochs for training"
    )
    parser.add_argument(
        "--sparsity_scale", type=float, default=1, help="Sparsity scale parameter"
    )

    args = parser.parse_args()
    main(args)
