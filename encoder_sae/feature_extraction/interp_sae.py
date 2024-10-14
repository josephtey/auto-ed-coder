import numpy as np

import os
import json
from datetime import datetime
import pickle
import sys

sys.path.append("../")
from shared.interp import run_interp_pipeline
from shared.sparse_autoencoder import SparseAutoencoder, SparseAutoencoderConfig
from shared.models import MiniPileDataset
import json
from dotenv import load_dotenv
import wandb
import uuid
import argparse


# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")


def interp_sae(
    sentences_file,
    embeddings_file,
    sae_base_path,
    features_base_path,
    max_features=None,
    model="gpt-4o-mini",
    feature_registry_path=None,
):
    # Start wandb run
    wandb.init(
        # set the wandb project where this run will be logged
        project="auto-ed-coder",
        # track hyperparameters and run metadata
        config={
            "type": "interp_sae",
            "sentences_file": sentences_file,
            "embeddings_file": embeddings_file,
            "sae_base_path": sae_base_path,
            "features_base_path": features_base_path,
            "max_features": max_features,
            "model": model,
        },
    )

    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_DIR = os.path.join(features_base_path, TIMESTAMP)

    # load dataset
    mini_pile_dataset = MiniPileDataset(sentences_file, embeddings_file)

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
    model = SparseAutoencoder(sae_config)
    model_path = os.path.join(sae_base_path, "sae.pkl")
    with open(model_path, "rb") as f:
        model_state_dict = pickle.load(f)
        model.load_state_dict(model_state_dict)

    # make folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    artifact = wandb.Artifact(name="sae-features-" + str(uuid.uuid4()), type="features")

    def write_labelled_feature_to_file(labelled_feature):
        feature_file_name = f"feature_{labelled_feature.index}.json"
        feature_path = os.path.join(OUTPUT_DIR, feature_file_name)
        with open(feature_path, "w") as json_file:
            json.dump(labelled_feature.dict(), json_file, indent=4)

        artifact.add_file(feature_path, feature_file_name)

    # Call the function
    run_interp_pipeline(
        model,
        mini_pile_dataset.embeddings,
        mini_pile_dataset.sentences,
        config["dimensions"] * 8,
        write_labelled_feature_to_file,
        max_features=max_features,
        model="gpt-4o-mini",
        feature_registry_path=feature_registry_path,
    )

    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpret Sparse Autoencoder Features"
    )

    parser.add_argument(
        "--sentences_file", type=str, required=True, help="Path to the sentences file"
    )
    parser.add_argument(
        "--embeddings_file", type=str, required=True, help="Path to the embeddings file"
    )

    parser.add_argument(
        "--sae_base_path",
        type=str,
        required=True,
        help="Base path for SAE config and model files",
    )
    parser.add_argument(
        "--features_base_path",
        type=str,
        required=True,
        help="OUTPUT base path to save the features",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=None,
        help="Maximum number of features to process",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for interpretation",
    )
    parser.add_argument(
        "--feature_registry_path",
        type=str,
        default=None,
        help="Path to a pre-computed feature registry file",
    )

    args = parser.parse_args()
    interp_sae(
        sentences_file=args.sentences_file,
        embeddings_file=args.embeddings_file,
        sae_base_path=args.sae_base_path,
        features_base_path=args.features_base_path,
        max_features=args.max_features,
        model=args.model,
        feature_registry_path=args.feature_registry_path,
    )
