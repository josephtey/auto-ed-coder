import argparse
import transformer_lens
import torch
import time

import sys
sys.path.append("../")

from shared import (
    SparseAutoencoder,
    SparseAutoencoderConfig,
    save_weights_with_description,
    load_weights_with_description
)
import torch.optim as optim
from datasets import load_dataset

import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix
from tqdm import tqdm

device = "cuda:0"
gpt_model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small").to(device)

embed_dims = 768

ds = load_dataset("JeanKaddour/minipile")

def train_sae(sae, weights_path = "./sae_weights.pth", description = None, max_iterations = 10000, mlp_layer = 5, batch_size = 1024):
  sae = SparseAutoencoder(SparseAutoencoderConfig(d_model=embed_dims, d_sparse=8 * embed_dims, sparsity_alpha=0)).to(device)
  optimizer = optim.Adam(sae.parameters(), lr=0.0001)

  sae.train()
  iteration = 0
  batch_count = 0
  losses = 0

  pbar = tqdm(range(max_iterations), desc="Processing")
  for sample in ds["train"]:

    # Uncomment if you want to turn the sparse factor up to 1 after 1/4 of the training process
    if iteration == int(max_iterations / 4):
      print("Changing alpha constant to 1")
      sae.config.sparsity_alpha = 1
    if iteration > max_iterations:
      break

    logits, activations = gpt_model.run_with_cache(sample["text"])

    mlp_out = activations.cache_dict[f"blocks.{mlp_layer}.hook_mlp_out"] # example MLP output, shape: (1, # samples, # dim)
    # print(mlp_out.shape)
    batch_count += mlp_out.shape[1]

    optimizer.zero_grad()
    y, f, loss, reconstruction_loss = sae(mlp_out, True)
    loss.backward()
    optimizer.step()

    losses += loss
    if batch_count > batch_size:
      # print(f"Loss: {losses / batch_count}")
      pbar.update(1)
      pbar.set_description(f"Loss: {losses / batch_count}")
      batch_count = 0
      losses = 0
      iteration += 1

  if description == None:
    description = time.time()
  save_weights_with_description(sae.state_dict(), description, weights_path)

  print(f"Saved weights to {weights_path}")

def test_sae(sae, weights_path = "./sae_weights.pth", max_samples = 100, mlp_layer = 5):
  sae = SparseAutoencoder(SparseAutoencoderConfig(d_model=embed_dims, d_sparse=8 * embed_dims, sparsity_alpha=0)).to(device)
  sae.eval()

  # Load in the SAE
  sae_weights = load_weights_with_description(weights_path)
  sae.load_state_dict(sae_weights)

  total_nonzero = 0
  nonzero_tokens = 0
  total_tokens = 0
  total = 0
  sample_cnt = 0
  pbar = tqdm(range(max_samples), desc = "Testing SAE")
  for sample in ds["test"]:
    sample_cnt += 1
    pbar.update(1)
    if sample_cnt > max_samples:
      break
    logits, activations = gpt_model.run_with_cache(sample["text"])

    mlp_out = activations.cache_dict[f"blocks.{mlp_layer}.hook_mlp_out"]

    y, f, loss, reconstruction_loss = sae(mlp_out, True)
    total_nonzero += torch.count_nonzero(f)
    total += f.shape[1] * f.shape[2] # num tokens * num dimensions
    nonzero_tokens += torch.count_nonzero(torch.count_nonzero(f, dim = 2)) # count the number of nonzero features per token, not just per paragraph sample
    total_tokens += f.shape[1]
  print(f"Nonzero: {total_nonzero}. Total: {total}. % nonzero features: {total_nonzero / total}")
  print(f"Nonzero tokens: {nonzero_tokens}. Total tokens: {total_tokens}. % nonzero tokens: {nonzero_tokens / total_tokens}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Example script with argument parser")

  parser.add_argument(
    '--mode',
    type=str,
    required=True,
    help='test or train mode'
  )

  parser.add_argument(
    '--weights_path',
    type=str,
    required=False,
    default="./sae_weights.pth",
    help="Path to save weights for train mode and load in weights for test mode"
  )

  parser.add_argument(
    '--description',
    type=str,
    required=False,
    help="Train mode: description for the weights that are being trained"
  )

  parser.add_argument(
    '--mlp_layer',
    type=int,
    required=False,
    default=5,
    help="Path to save weights for train mode and load in weights for test mode"
  )

  parser.add_argument(
    '--max_samples',
    type=int,
    required=False,
    default=100,
    help="Test mode: number of samples in the dataset to evaluate on"
  )

  parser.add_argument(
    '--max_iterations',
    type=int,
    required=False,
    default=1000,
    help="Train mode: number of batches to train on"
  )

  parser.add_argument(
    '--batch_size',
    type=int,
    required=False,
    default=1024,
    help="Train mode: size of minimum batch"
  )

  args = parser.parse_args()

  if args.mode == "train":
    train_sae(sae, description=args.description, weights_path=args.weights_path, mlp_layer=args.mlp_layer, max_iterations=args.max_iterations, batch_size=args.batch_size)
  elif args.mode == "test":
    test_sae(sae, weights_path=args.weights_path, mlp_layer=args.mlp_layer, max_samples=args.max_samples)
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