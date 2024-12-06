import torch.optim as optim
import torch

import sys

sys.path.append("../")
from shared.sparse_autoencoder import SparseAutoencoder, SparseAutoencoderConfig, SparseAutoencoderType
from shared.models import MiniPileDataset

from datetime import datetime
import pickle
import json
import os
import argparse
import numpy as np
from tqdm import tqdm
import uuid


def train_sae(
    sentences_file,
    embeddings_file,
    run_folder,
    batch_size,
    dimensions,
    sparsity_alpha,
    lr,
    num_epochs,
    sparsity_scale,
    wandb,
    sae_type,
    top_k,
    tie_weights,
):
    # Load sentences and embeddings into MiniPileDataset
    dataset = MiniPileDataset(sentences_file, embeddings_file)

    os.makedirs(run_folder, exist_ok=True)

    # Config
    config = {
        "batch_size": batch_size,
        "dimensions": dimensions,
        "sparsity_alpha": sparsity_alpha,
        "lr": lr,
        "num_epochs": num_epochs,
        "sparsity_scale": sparsity_scale,
        "tie_weights": tie_weights,
    }

    # Assuming `dataset` is a PyTorch Dataset loaded and ready to use
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True
    )

    # Initialize the model
    sae_config = SparseAutoencoderConfig(
        d_model=config["dimensions"],
        d_sparse=8 * config["dimensions"],
        sparsity_alpha=config["sparsity_alpha"],
        top_k = top_k,
        sae_type = sae_type,
        tie_weights = tie_weights,
    )
    model = SparseAutoencoder(sae_config)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # Training loop
    with open(f"{run_folder}/training_output.out", "w") as out_file:
        num_epochs = config["num_epochs"]
        log_feature_densities = []
        training_feature_activations = []
        for epoch in range(num_epochs):
            total_loss = 0
            batch_feature_activations = []
            for sentences, embeddings in tqdm(data_loader, desc=f"Epoch {epoch+1}"):
                optimizer.zero_grad()

                # Assuming data is already on the correct device and in the correct format
                # Feature activations: (batch size, # features)
                _, feature_activations, loss, _ = model.forward(
                    embeddings,
                    return_loss=True,
                    sparsity_scale=config["sparsity_scale"],
                    new_loss=False,
                )
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

                batch_feature_activations.append(feature_activations)

                # Print the loss for every batch
                batch_loss_str = f"Batch Loss: {loss.item()}"
                wandb.log({"batch_loss": loss.item()})
                out_file.write(batch_loss_str + "\n")

            batch_feature_activations = torch.concat(batch_feature_activations, dim = 0)

            # Calculate log feature densities
            log_feature_density = np.log(
                (
                    (batch_feature_activations.detach().numpy() > 0).sum(axis=0)
                    / batch_feature_activations.shape[0]
                )
            )

            log_feature_densities.append(log_feature_density.tolist())

            epoch_loss_str = (
                f"Epoch {epoch+1}, Average Loss: {total_loss / len(data_loader)}"
            )
            wandb.log(
                {"epoch": epoch + 1, "average_loss": total_loss / len(data_loader)}
            )
            out_file.write(epoch_loss_str + "\n")
            print(epoch_loss_str)

    # Save final loss
    wandb.log({"average_loss": total_loss / len(data_loader)})

    # Save the model to a pickle file
    model_path = f"{run_folder}/sae.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model.state_dict(), f)

    print(f"SAE saved to {model_path}")

    # Save configuration
    config_path = f"{run_folder}/config.json"
    with open(config_path, "w") as config_file:
        json.dump(config, config_file)

    print(f"Configuration saved to {config_path}")

    # Save log feature densities to a JSON file
    log_feature_densities_path = f"{run_folder}/log_feature_densities.json"
    with open(log_feature_densities_path, "w") as json_file:
        json.dump(log_feature_densities, json_file)

    print(f"Log feature densities saved to {log_feature_densities_path}")

    # Save all of this into an artifact
    artifact = wandb.Artifact(name="sae-" + str(uuid.uuid4()), type="model")
    artifact.add_file(model_path, "SAE Model")
    artifact.add_file(config_path, "SAE Config")
    artifact.add_file(log_feature_densities_path, "Log Feature Densities")
    wandb.log_artifact(artifact)

    print(f"Artifact saved to {artifact}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder")
    parser.add_argument(
        "--sentences_file", type=str, required=True, help="Path to the sentences file"
    )
    parser.add_argument(
        "--embeddings_file", type=str, required=True, help="Path to the embeddings file"
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

    parser.add_argument(
        "--sae_type", type=SparseAutoencoderType, default=SparseAutoencoderType.BASIC, help="Type of Sparse Autoencoder"
    )

    parser.add_argument(
        "--top_k", type=int, default=100, help="Top k value (only used for top k sparse autoencoder)"
    )

    args = parser.parse_args()
    train_sae(
        sentences_file=args.sentences_file,
        embeddings_file=args.embeddings_file,
        run_folder=args.run_folder,
        batch_size=args.batch_size,
        dimensions=args.dimensions,
        sparsity_alpha=args.sparsity_alpha,
        lr=args.lr,
        num_epochs=args.num_epochs,
        sparsity_scale=args.sparsity_scale,
        sae_type=args.sae_type,
        top_k=args.top_k,
    )
