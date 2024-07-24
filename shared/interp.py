"""
This module provides functionality to interpret the features learned by a Sparse Autoencoder (SAE) model.
It uses OpenAI's API to generate human-readable interpretations of the features and scores their relevance.

Classes:
    OpenAIClient: A client to interact with OpenAI's API for generating and scoring interpretations.

Functions:
    run_interp_pipeline: Main function to run the interpretation pipeline. It processes the embeddings, 
                         generates feature activations, and uses OpenAI's API to interpret and score the features.

Usage:
    1. Ensure you have a .env file with your OpenAI API key.
    2. Call the run_interp_pipeline function with the appropriate parameters:
        - sae: The Sparse Autoencoder model.
        - embeddings: The embeddings to be processed.
        - text_data: The corresponding text data for the embeddings.
        - n_feature_activations: Number of feature activations to consider.
        - handle_labelled_feature: A callback function to handle the labelled feature.

Example:
    from shared.sparse_autoencoder import SparseAutoencoder, SparseAutoencoderConfig
    from shared.models import MiniPileDataset
    import pickle

    # Load the dataset
    with open("path_to_dataset.pkl", "rb") as f:
        mini_pile_dataset = pickle.load(f)

    # Load the pre-trained model
    sae_config = SparseAutoencoderConfig(d_model=768, d_sparse=6144, sparsity_alpha=1)
    model = SparseAutoencoder(sae_config)
    with open("path_to_model.pkl", "rb") as f:
        model_state_dict = pickle.load(f)
        model.load_state_dict(model_state_dict)

    # Define a callback function to handle the labelled feature
    def handle_labelled_feature(labelled_feature):
        print(labelled_feature)

    # Run the interpretation pipeline
    run_interp_pipeline(
        model,
        mini_pile_dataset.embeddings,
        mini_pile_dataset.sentences,
        6144 (usually n_dimensions * 8),
        handle_labelled_feature
    )
"""

import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

from shared.ai import OpenAIClient
from shared.sparse_autoencoder import SparseAutoencoder, SparseAutoencoderConfig
from dotenv import load_dotenv
from shared.features import Feature, FeatureSample
from shared.models import MiniPileDataset
import tqdm
import os

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")


def run_interp_pipeline(
    sae, embeddings, text_data, n_feature_activations, handle_labelled_feature
):
    ai = OpenAIClient(openai_api_key)

    n = len(embeddings)
    feature_registry = np.zeros((n_feature_activations, n))

    for i in tqdm.tqdm(range(n), desc="Creating feature registry"):
        embedding = torch.tensor(embeddings[i])
        feature_activations = sae.forward(embedding)[1]
        feature_registry[:, i] = feature_activations.detach().numpy()

    for index, feature in enumerate(
        tqdm.tqdm(feature_registry, desc="Processing features")
    ):
        feature_samples = [
            FeatureSample(text=text_data[i], act=value)
            for i, value in enumerate(feature)
        ]
        feature_samples.sort(key=lambda x: x.act, reverse=True)

        high_act_samples = [
            sample for sample in feature_samples[:50] if sample.act != 0
        ]
        low_act_samples = np.random.choice(
            [sample for sample in feature_samples if sample.act == 0], 50, replace=False
        ).tolist()

        try:
            interpetation = ai.get_interpretation(high_act_samples, low_act_samples)
            label = interpetation["label"]
            reasoning = interpetation["reasoning"]
            attributes = interpetation["attributes"]

            high_act_score = ai.score_interpretation(high_act_samples, attributes)[
                "percent"
            ]
            low_act_score = ai.score_interpretation(low_act_samples, attributes)[
                "percent"
            ]
        except Exception as e:
            print(f"Skipping feature due to error: {e}")
            continue

        labelled_feature = Feature(
            index=index,
            label=label,
            attributes=attributes,
            reasoning=reasoning,
            confidence=abs(high_act_score - low_act_score),
            density=(np.count_nonzero(feature) / len(feature)),
            high_act_samples=high_act_samples,
            low_act_samples=low_act_samples,
        )

        handle_labelled_feature(labelled_feature)


def plot_feature_activation_histogram(model, mini_pile_dataset, num_samples=1000):
    # Initialize a dictionary to count non-zero activations for each feature
    feature_non_zero_count = defaultdict(int)
    num_samples = min(num_samples, len(mini_pile_dataset.embeddings))

    # Count non-zero activations for each feature across samples
    for sample_idx in range(num_samples):
        embedding = torch.tensor(mini_pile_dataset.embeddings[sample_idx])
        feature_activations = model.forward(embedding)[1]
        non_zero_indices = torch.nonzero(feature_activations, as_tuple=True)[0]
        for idx in non_zero_indices:
            feature_non_zero_count[idx.item()] += 1

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.bar(
        feature_non_zero_count.keys(), feature_non_zero_count.values(), color="skyblue"
    )
    plt.xlabel("Feature Index", fontsize=14)
    plt.ylabel("Non-Zero Count", fontsize=14)
    plt.title(f"Feature Non-Zero Count for first {num_samples} samples", fontsize=16)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()


def count_non_zero_feature_activations(model, mini_pile_dataset, num_samples=100):
    total_non_zero = 0
    total_elements = 0
    percent_count = 0
    num_samples = min(num_samples, len(mini_pile_dataset.embeddings))

    for sample_idx in range(num_samples):
        embedding = torch.tensor(mini_pile_dataset.embeddings[sample_idx])
        feature_activations = model.forward(embedding)[1]
        non_zero_elements = torch.count_nonzero(feature_activations)
        total_non_zero += non_zero_elements
        total_elements += feature_activations.numel()

        percent_count += non_zero_elements / feature_activations.numel()

    average_non_zero = total_non_zero / num_samples
    percentage_non_zero = (percent_count / num_samples) * 100

    print(
        f"Average Non-Zero Elements for first {num_samples} samples: {average_non_zero}"
    )
    print(f"Average Percentage of Non-Zero Elements: {percentage_non_zero:.2f}%")
