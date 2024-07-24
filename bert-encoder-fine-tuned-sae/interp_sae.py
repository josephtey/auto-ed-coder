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

import argparse


# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")


def main(args):
    # load dataset
    mini_pile_dataset = MiniPileDataset(args.sentences_file, args.embeddings_file)

    # Load the configuration from the JSON file
    config_path = os.path.join(args.sae_base_path, "config.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    # Load the pre-trained model from the pickle file
    sae_config = SparseAutoencoderConfig(
        d_model=config["dimensions"],
        d_sparse=8 * config["dimensions"],
        sparsity_alpha=config["sparsity_alpha"],
    )
    model = SparseAutoencoder(sae_config)
    model_path = os.path.join(args.sae_base_path, "sae.pkl")
    with open(model_path, "rb") as f:
        model_state_dict = pickle.load(f)
        model.load_state_dict(model_state_dict)

    # make folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.join(args.features_base_path, f"sae_features_{timestamp}")
    os.makedirs(folder_name, exist_ok=True)

    def write_labelled_feature_to_file(labelled_feature):
        with open(
            os.path.join(folder_name, f"feature_{labelled_feature.index}.json"), "w"
        ) as json_file:
            json.dump(labelled_feature.dict(), json_file, indent=4)

    # Call the function
    run_interp_pipeline(
        model,
        mini_pile_dataset.embeddings,
        mini_pile_dataset.sentences,
        config["dimensions"] * 8,
        write_labelled_feature_to_file,
        max_features=args.max_features,
    )


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

    args = parser.parse_args()
    main(args)
