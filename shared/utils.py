import torch
import matplotlib.pyplot as plt
import numpy as np
import requests
import json
from datetime import datetime
import nltk

nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize
import torch


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


def build_heatmap_using_encoder_method(
    text, sae_model, model_type, output_folder, first_n_features
):

    # 1. Split the text into sentences
    sentences = sent_tokenize(text)

    # 2. Embed each sentence
    embeddings = []
    for sentence in sentences:
        if model_type == "fine-tuned":
            response = requests.post(
                "https://josephtey--bert-set-3-fine-tuned.modal.run",
                json={"text": sentence},
            )
        else:
            response = requests.post(
                "https://josephtey--bert-base-uncased.modal.run",
                json={"text": sentence},
            )
        embedding = response.json()["embedding"]
        embeddings.append(embedding)

    # 3. Get feature activations
    feature_activations = []
    for embedding in embeddings:
        embedding_tensor = torch.tensor(embedding)
        activation = sae_model.forward(embedding_tensor)[1]
        feature_activations.append(activation[:first_n_features].tolist())

    # 4. Write to JSON file
    output = []
    for sentence, embedding, activation in zip(
        sentences, embeddings, feature_activations
    ):
        output.append(
            {
                "sentence": sentence,
                "embedding": embedding,
                "feature_activations": activation,
            }
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_folder}/heatmap_data_{model_type}_{timestamp}.json"

    with open(filename, "w") as f:
        json.dump(output, f)

    # 5. Return the output
    return output

