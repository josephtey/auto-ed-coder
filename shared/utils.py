import torch
import matplotlib.pyplot as plt
import numpy as np


def save_weights_with_description(weights, description, path):
    torch.save({"weights": weights, "description": description}, path)


def load_weights_with_description(path, print_description=True):
    saved_weights = torch.load(path)
    assert "weights" in saved_weights and "description" in saved_weights

    if print_description:
        print(f"Loaded weights with description: {saved_weights['description']}")

    return saved_weights["weights"]


def log_feature_density_histogram(
    f, bins=50, min_log=-20, out_file="./fd_histogram.png"
):
    # feature_activations: (# batches, batch size, activation dimension)
    log_feature_density = (
        torch.log10(torch.count_nonzero(f, dim=1) / f.shape[1]).to("cpu").numpy()
    )

    # Convert into a heatmap that has log density on y-axis, training steps on x-axis, and heatmap value being the distribution
    heatmap = []
    for training_step in log_feature_density:
        # print(training_step)
        binned_vals, _ = np.histogram(training_step, bins=bins, range=(min_log, 0))

        # collect all values below the minimum threshold and put them in one bucket
        vals_under_min = training_step < min_log
        binned_vals = np.insert(binned_vals, 0, vals_under_min.sum())

        heatmap.append(binned_vals)

    res = np.stack(heatmap, axis=0)

    plt.figure(figsize=(8, 6))
    plt.imshow(res.T, aspect="auto", cmap="viridis", origin="lower")
    plt.ylabel("Log feature density from -20 (bottom) to 0 (top)")
    plt.xlabel(f"Num batches (up to {f.shape[0]})")
    plt.yticks(ticks=[])
    plt.xticks(ticks=[])

    plt.colorbar(label="Count")

    plt.savefig(out_file)
