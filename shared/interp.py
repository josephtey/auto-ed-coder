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
        - feature_registry_path: Optional path to a pre-computed feature registry file.

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
        handle_labelled_feature,
        feature_registry_path="path_to_feature_registry.npy"
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


import ipywidgets as widgets
from IPython.display import display, HTML
import matplotlib.colors as mcolors
import re
from heapq import nlargest


# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")


def run_interp_pipeline(
    sae,
    embeddings,
    text_data,
    n_feature_activations,
    handle_labelled_feature,
    max_features=None,
    model="gpt-4o-mini",
    feature_registry_path=None,
):
    ai = OpenAIClient(openai_api_key, model=model)

    n = len(embeddings)

    if feature_registry_path and os.path.exists(feature_registry_path):
        print(f"Loading feature registry from {feature_registry_path}")
        feature_registry = np.load(feature_registry_path, mmap_mode="r")
    else:
        print("Creating feature registry")
        feature_registry = np.zeros((n_feature_activations, n))

        for i in tqdm.tqdm(range(n), desc="Creating feature registry"):
            embedding = torch.tensor(embeddings[i])
            feature_activations = sae.forward(embedding)[1]
            feature_registry[:, i] = feature_activations.detach().numpy()

        if feature_registry_path:
            np.save(feature_registry_path, feature_registry)
            print(f"Feature registry saved to {feature_registry_path}")

    if max_features is not None:
        feature_registry = feature_registry[:max_features, :]

    for index, feature in enumerate(
        tqdm.tqdm(feature_registry, desc="Processing features")
    ):
        feature_samples = [
            FeatureSample(text=text_data[i], act=value)
            for i, value in enumerate(feature)
        ]
        feature_samples.sort(key=lambda x: x.act, reverse=True)

        # Get high activation samples
        high_act_samples = nlargest(
            50,
            (sample for sample in feature_samples if sample.act > 0),
            key=lambda x: x.act,
        )

        # Get low activation samples
        low_act_samples_population = [
            sample for sample in feature_samples if sample.act == 0
        ]
        num_low_act_samples = min(50, len(low_act_samples_population))
        low_act_samples = np.random.choice(
            low_act_samples_population, num_low_act_samples, replace=False
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


def plot_feature_activation_histogram_from_log_feature_densities(
    log_feature_densities, training_step
):
    feature_activations = log_feature_densities[training_step]

    # Filter out -inf values
    filtered_feature_activations = [
        fa for fa in feature_activations if fa != -float("inf")
    ]
    print("len(filtered_feature_activations)", len(filtered_feature_activations))

    # Plotting the histogram of feature activations
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_feature_activations, bins=30, edgecolor="black", alpha=0.7)
    plt.xlabel("Log Feature Densities", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title(f"Histogram of Feature Activations at Step {training_step}", fontsize=16)
    plt.grid(True)
    plt.show()


def plot_highest_activating_feature_for_each_sentence(
    fine_tuned_heatmap_data,
    fine_tuned_features,
    comparison_heatmap_data=None,
    comparison_features=None,
):
    print("Analyzing feature activations for each sentence:")
    print("=" * 80)

    for idx, entry in enumerate(fine_tuned_heatmap_data):
        print(f"\nSentence {idx + 1}: {entry['sentence']}")
        print("-" * 80)

        # Create a figure with two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

        # Function to plot and print feature information
        def plot_and_print_features(ax, heatmap_data, features, title):
            activations = heatmap_data[idx]["feature_activations"]
            top_5_indices = sorted(
                range(len(activations)), key=lambda i: activations[i], reverse=True
            )[:5]
            top_5_activations = [activations[i] for i in top_5_indices]
            top_5_labels = []
            for feature_idx in top_5_indices:
                matching_feature = next(
                    (f for f in features if f.index == feature_idx), None
                )
                label = (
                    matching_feature.label
                    if matching_feature
                    else f"Feature {feature_idx}"
                )
                top_5_labels.append(label)

            y_pos = np.arange(len(top_5_labels))
            ax.barh(y_pos, top_5_activations)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_5_labels)
            ax.set_title(title)
            ax.set_xlabel("Activation Strength")
            ax.set_ylabel("Feature Label")
            ax.set_xlim(0, 1)

            for i, v in enumerate(top_5_activations):
                ax.text(v, i, f" {v:.4f}", va="center")

            print(f"Top 5 activating features for {title}:")
            for feature_idx, activation in zip(top_5_indices, top_5_activations):
                matching_feature = next(
                    (f for f in features if f.index == feature_idx), None
                )
                if matching_feature:
                    print(
                        f"  Feature {feature_idx:4d} | Activation: {activation:.4f} | Label: {matching_feature.label}"
                    )
                else:
                    print(
                        f"  Feature {feature_idx:4d} | Activation: {activation:.4f} | Label: N/A"
                    )
            print()

        # Plot and print for fine-tuned model
        plot_and_print_features(
            ax1, fine_tuned_heatmap_data, fine_tuned_features, "Fine-tuned Model"
        )

        # Plot and print for comparison model if provided
        if comparison_heatmap_data and comparison_features:
            plot_and_print_features(
                ax2, comparison_heatmap_data, comparison_features, "Pre-Trained Model"
            )
        else:
            ax2.axis("off")

        plt.tight_layout()
        plt.show()

    print("=" * 80)
    print("Analysis complete.")


def create_feature_heatmap_widget(fine_tuned_features, fine_tuned_heatmap_data):
    def create_heatmap(feature_index, heatmap_data):
        feature = next(
            (f for f in fine_tuned_features if f.index == feature_index), None
        )
        if not feature:
            return HTML("Feature not found.")

        # Normalize activations
        activations = [
            entry["feature_activations"][feature_index] for entry in heatmap_data
        ]
        norm = mcolors.Normalize(vmin=min(activations), vmax=max(activations))

        # Create HTML for heatmap
        html_content = f"<h3>Heatmap for Feature {feature_index}: {feature.label}</h3>"

        for entry in heatmap_data:
            sentence = entry["sentence"]
            activation = entry["feature_activations"][feature_index]
            words = re.findall(r"\S+|\s+", sentence)
            for word in words:
                # Use shades of orange, darker for stronger activation
                color = mcolors.to_hex(plt.cm.Oranges(norm(activation)))
                html_content += (
                    f"<span style='background-color: {color};'>{word}</span>"
                )

        html_content += "</p></div>"

        # Add feature details
        html_content += "<div style='margin-bottom: 20px;'>"
        html_content += f"<p><strong>Attributes:</strong> {feature.attributes}</p>"
        html_content += f"<p><strong>Reasoning:</strong> {feature.reasoning}</p>"
        html_content += "<h4>High Acting Samples:</h4>"
        for sample in feature.high_act_samples[:5]:  # Show first 5 high acting samples
            html_content += f"<p>'{sample.text}' (Activation: {sample.act:.4f})</p>"
        html_content += "<h4>Low Acting Samples:</h4>"
        for sample in feature.low_act_samples[:5]:  # Show first 5 low acting samples
            html_content += f"<p>'{sample.text}' (Activation: {sample.act:.4f})</p>"
        html_content += "</div>"

        html_content += "<div style='width: 100%; overflow-x: auto;'>"
        html_content += "<p style='line-height: 1.5;'>"

        return HTML(html_content)

    # Create search widget
    search_box = widgets.Text(
        value="",
        placeholder="Search features...",
        description="Search:",
        disabled=False,
    )

    # Create dropdown widget
    feature_dropdown = widgets.Dropdown(
        options=[(f"{f.index}: {f.label}", f.index) for f in fine_tuned_features[:500]],
        description="Feature:",
        style={"description_width": "initial"},
        layout={"width": "max-content"},
    )

    # Create output widget
    output = widgets.Output()

    # Define function to update output when dropdown value changes
    def on_feature_change(change):
        with output:
            output.clear_output()
            display(create_heatmap(change.new, fine_tuned_heatmap_data))

    # Define function to update dropdown options based on search
    def on_search_change(change):
        search_term = change.new.lower()
        filtered_features = [
            (f"{f.index}: {f.label}", f.index)
            for f in fine_tuned_features[:500]
            if search_term in f.label.lower() or search_term in str(f.index)
        ]
        feature_dropdown.options = filtered_features

    # Connect the dropdown to the update function
    feature_dropdown.observe(on_feature_change, names="value")

    # Connect the search box to the search function
    search_box.observe(on_search_change, names="value")

    # Display the search box, dropdown and output
    display(search_box, feature_dropdown, output)

    # Initialize with the first feature
    with output:
        display(create_heatmap(fine_tuned_features[0].index, fine_tuned_heatmap_data))


# Usage:
# create_feature_heatmap_widget(fine_tuned_features, fine_tuned_heatmap_data)
