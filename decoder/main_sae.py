import argparse
import transformer_lens
import torch
import time

import sys

sys.path.append("../")
from shared.utils import LogFeatureDensityHistogram, get_datetime_string

from shared import (
    SparseAutoencoder,
    SparseAutoencoderConfig,
    save_weights_with_description,
    load_weights_with_description,
)
import torch.optim as optim
from datasets import load_dataset

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from mini_pile_ds import MiniPileDataset
from torch.utils.data import DataLoader
import os

device = "cuda:0"
# gpt_model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small").to(device)

embed_dims = 768

def generate_description(max_batch_size, batch_size, sparsity, lr):
  return f"Model trained at time {get_datetime_string()} with parameters:\nbatch_size = {batch_size},\nmax_batch_size = {max_batch_size},\nsparsity = {sparsity},\nlr = {lr}"

def train_sae(weights_name = "sae_weights", description = None, max_batch_count = 10000, batch_size = 1024, sparsity_alpha = 1, lr = 0.001):
  # Initialize where we'll store weights and data results
  weights_path = os.path.join("./weights", weights_name + ".pth")
  # weights_path = os.path.join(weights_dir, weights_name)
  # os.mkdir(weights_dir)

  ds = MiniPileDataset("./mini_pile_train.npy", device = device)
  sae = SparseAutoencoder(SparseAutoencoderConfig(d_model=embed_dims, d_sparse=8 * embed_dims, sparsity_alpha=sparsity_alpha)).to(device)
  optimizer = optim.Adam(sae.parameters(), lr=lr)

  sae.train()
  max_batch_count = float("inf") if max_batch_count == None else max_batch_count
  batch_count = 0
  losses = 0

  dataloader = DataLoader(ds, batch_size=batch_size, shuffle=True)

  pbar = tqdm(range(min(max_batch_count, len(ds))), desc="Processing")
  log_histogram = LogFeatureDensityHistogram()
  for sample in dataloader:

    # Uncomment if you want to turn the sparse factor up to 1 after 1/4 of the training process
    # if batch_count == int(len(dataloader / 4)):
    #   print("Changing alpha constant to 1")
    #   sae.config.sparsity_alpha = 100
    if batch_count > max_batch_count:
      break

    optimizer.zero_grad()
    y, f, loss, reconstruction_loss = sae(sample, True)
    loss.backward()
    optimizer.step()
    losses += loss

    # print(f"Loss: {losses / batch_count}")
    pbar.update(1)
    pbar.set_description(f"Loss: {loss}")
    batch_count += 1
    y, f, loss, reconstruction_loss = sae(sample.unsqueeze(0), True)

    log_histogram.add_feature_activation(f)

    if batch_count % 100 == 0:
      log_histogram.make_histogram(out_file=f"./data/fd{batch_count}.png")

  if description == None:
    description = generate_description(max_batch_count, batch_size, sparsity_alpha, lr)
  save_weights_with_description(sae.state_dict(), description, weights_path)

  print(f"Saved weights to {weights_path}")

def test_sae(weights_name = "sae_weights", max_samples = 100):
  # Find our weights directory
  ds = MiniPileDataset("./mini_pile_test.npy", device = device)
  dataloader = DataLoader(ds, batch_size=1024, shuffle=True)

  sae = SparseAutoencoder(SparseAutoencoderConfig(d_model=embed_dims, d_sparse=8 * embed_dims, sparsity_alpha=0)).to(device)
  sae.eval()

  # Load in the SAE
  sae_weights = load_weights_with_description(os.path.join("./weights", weights_name + ".pth"))
  sae.load_state_dict(sae_weights)

  total_nonzero = 0
  nonzero_tokens = 0
  total_tokens = 0
  total = 0
  sample_cnt = 0
  pbar = tqdm(range(min(max_samples, len(ds))), desc = "Testing SAE")
  feature_nonzeros = torch.zeros((8 * embed_dims,)).to(device)
  for sample in dataloader:
    # sample: (batch_size, feature dimensions)
    sample_cnt += 1
    pbar.update(1)

    if sample_cnt > max_samples:
      break

    y, f, loss, reconstruction_loss = sae(sample.unsqueeze(0), True)
    # Break down nonzero scores by feature
    feature_nonzeros += torch.count_nonzero(f, dim = 1).squeeze()

    total_nonzero += torch.count_nonzero(f)
    total += f.shape[1] * f.shape[2] # num tokens * num dimensions
    nonzero_tokens += torch.count_nonzero(torch.count_nonzero(f, dim = 2)) # count the number of nonzero features per token, not just per paragraph sample
    total_tokens += f.shape[1]
  print(f"Dead features: {(feature_nonzeros == 0).sum()}")
  print(f"Nonzero: {total_nonzero}. Total: {total}. % nonzero features: {100 * total_nonzero / total}%")
  print(f"Nonzero tokens: {nonzero_tokens}. Total tokens: {total_tokens}. % nonzero tokens: {100 * nonzero_tokens / total_tokens}%")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Example script with argument parser")

  parser.add_argument(
    '--mode',
    type=str,
    required=True,
    help='test or train mode'
  )

  parser.add_argument(
    '--weights_name',
    type=str,
    required=False,
    default="sae_weights",
    help="Path to save weights for train mode and load in weights for test mode"
  )

  parser.add_argument(
    '--description',
    type=str,
    required=False,
    help="Train mode: description for the weights that are being trained"
  )

  parser.add_argument(
    '--max_samples',
    type=int,
    required=False,
    default=100,
    help="Test mode: number of samples in the dataset to evaluate on"
  )

  parser.add_argument(
    '--max_batch_count',
    type=int,
    required=False,
    help="Train mode: number of batches to train on"
  )

  parser.add_argument(
    '--batch_size',
    type=int,
    required=False,
    default=1024,
    help="Train mode: size of minimum batch"
  )

  parser.add_argument(
    '--sparsity',
    type=float,
    required=False,
    default=1,
    help="Train mode: sparsity constraint"
  )

  parser.add_argument(
    '--lr',
    type=float,
    required=False,
    default=0.001,
    help="Train mode: sparsity constraint"
  )

  args = parser.parse_args()

  if args.mode == "train":
    train_sae(description=args.description, weights_name=args.weights_name, max_batch_count=args.max_batch_count, batch_size=args.batch_size)
  elif args.mode == "test":
    test_sae(weights_name=args.weights_name, max_samples=args.max_samples)
  else:
    raise Exception(f"Mode {args.mode} is invalid")



### FOR ABLATIONS
# layer_to_ablate = 0
# head_index_to_ablate = 8

# # We define a head ablation hook
# # The type annotations are NOT necessary, they're just a useful guide to the reader
# #
# def head_ablation_hook(
#     value,
#     hook
# ):
#     print(f"Shape of the value tensor: {value.shape}")
#     value[:, :, head_index_to_ablate, :] = 0.
#     return value
# gpt2_tokens = gpt_model.to_tokens("Hello world!")
# original_loss = gpt_model(gpt2_tokens, return_type="loss")
# ablated_loss = gpt_model.run_with_hooks(
#     gpt2_tokens,
#     return_type="loss",
#     fwd_hooks=[(
#         utils.get_act_name("v", layer_to_ablate),
#         head_ablation_hook
#         )]
#     )

# print(activations.get_neuron_results(4))