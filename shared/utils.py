import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def save_weights_with_description(weights, description, path):
  torch.save({
    "weights": weights,
    "description": description
  }, path)

def load_weights_with_description(path, print_description = True):
  saved_weights = torch.load(path)
  assert "weights" in saved_weights and "description" in saved_weights

  if print_description:
    print(f"Loaded weights with description: {saved_weights['description']}")

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