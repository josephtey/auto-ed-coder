import torch
from collections import defaultdict
import matplotlib.pyplot as plt


def plot_feature_activation_histogram(model, mini_pile_dataset, num_samples=1000):
    # Initialize a dictionary to count non-zero activations for each feature
    feature_non_zero_count = defaultdict(int)
    num_samples = min(num_samples, len(mini_pile_dataset.embeddings))

    # Count non-zero activations for each feature across samples
    for sample_idx in range(num_samples):
        feature_activations = model.forward(mini_pile_dataset.embeddings[sample_idx])[1]
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
        feature_activations = model.forward(mini_pile_dataset.embeddings[sample_idx])[1]
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
