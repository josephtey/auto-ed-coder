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

    return saved_weights["weights"]

class LogFeatureDensityHistogram():
  def __init__(self, bins = 50, min_log = -20):
    self.bins = bins
    self.min_log = min_log
    self.heatmap = []

  def add_feature_activation(self, f):
    # feature activation shape: (1, batch size, activation dimension)
    log_feature_density = torch.log10(torch.count_nonzero(f, dim = 1) / f.shape[1]).to("cpu").numpy()

    binned_vals, _ = np.histogram(log_feature_density.squeeze(), bins=self.bins, range=(self.min_log, 0))

    vals_under_min = (log_feature_density.squeeze() < self.min_log)
    binned_vals = np.insert(binned_vals, 0, vals_under_min.sum())

    self.heatmap.append(binned_vals)

  def make_histogram(self, out_file = "./fd_histogram.png"):
    heatmap = np.stack(self.heatmap, axis=0)

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap.T, aspect='auto', cmap='viridis', origin='lower')
    plt.ylabel("Log feature density from -20 (bottom) to 0 (top)")
    plt.xlabel(f"Num batches (up to {heatmap.shape[0]})")
    plt.yticks(ticks=[])
    plt.xticks(ticks=[])

    plt.colorbar(label='Count')

    plt.savefig(out_file)

def get_datetime_string():
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    return date_string

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
