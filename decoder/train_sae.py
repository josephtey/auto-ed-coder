import transformer_lens
import torch

import sys
sys.path.append("../")

from shared import (
    BottleneckT5Autoencoder,
    SparseAutoencoder,
    SparseAutoencoderConfig,
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
model = transformer_lens.HookedTransformer.from_pretrained("gpt2-small").to(device)
# Run the model and get logits and activations
logits, activations = model.run_with_cache("Hello World")

layer = 5
mlp_out = activations.cache_dict[f"blocks.{layer}.hook_mlp_out"] # example MLP output

embed_dims = 768
sae = SparseAutoencoder(SparseAutoencoderConfig(d_model=embed_dims, d_sparse=8 * embed_dims, sparsity_alpha=1)).to(device)
optimizer = optim.Adam(sae.parameters(), lr=0.001)

ds = load_dataset("JeanKaddour/minipile")


model.train()
iteration = 0
mlps = None
num_samples = 10000
batch_size = 500

for sample in ds["train"]:
  pbar = tqdm(range(batch_size), desc="Processing")
  losses = 0

  if iteration > num_samples:
    break

  for i in range(batch_size):
    iteration += 1
    pbar.update(1)

    logits, activations = model.run_with_cache(sample["text"])

    layer = 5
    mlp_out = activations.cache_dict[f"blocks.{layer}.hook_mlp_out"] # example MLP output

    # if mlps == None:
    #   mlps = mlp_out
    # else:
    #   mlps = torch.cat([mlps, mlp_out], dim = 1)

  # torch.save(mlps, "sae_data.pt")

    optimizer.zero_grad()
    y, f, loss, reconstruction_loss = sae(mlp_out, True)
    loss.backward()
    optimizer.step()

    losses += loss
  print(f"Loss: {losses / batch_size}")

torch.save(model.state_dict(), "sea_weights.pth")





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
# gpt2_tokens = model.to_tokens("Hello world!")
# original_loss = model(gpt2_tokens, return_type="loss")
# ablated_loss = model.run_with_hooks(
#     gpt2_tokens,
#     return_type="loss",
#     fwd_hooks=[(
#         utils.get_act_name("v", layer_to_ablate),
#         head_ablation_hook
#         )]
#     )

# print(activations.get_neuron_results(4))