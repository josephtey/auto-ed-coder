from datasets import load_dataset
import transformer_lens
from tqdm import tqdm
import random

import argparse
import transformer_lens
import torch
import time

import os
import sys

sys.path.append("../")
from shared.utils import LogFeatureDensityHistogram, get_datetime_string

from shared import (
    SparseAutoencoder,
    SparseAutoencoderConfig,
    save_weights_with_description,
    load_weights_with_description,
)

device = "cuda:0"
device1 = "cuda:0"

embed_dims = 768
sae = SparseAutoencoder(SparseAutoencoderConfig(d_model=embed_dims, d_sparse=8 * embed_dims, sparsity_alpha=0)).to(device1)

sae_weights = load_weights_with_description(os.path.join("./weights", "sae_weights.pth"))
sae.load_state_dict(sae_weights)

def prune_stored_sentences(sentences, top_k = 50):
  # assume sentences is (feature activation, sentence)
  # Keep the lowest and highest top_k feature activations
  random.shuffle(sentences)
  sentences.sort(key = lambda ele: ele[0]) # only consider the feature activation when sorting

  if len(sentences) < 2 * top_k:
    return sentences

  new_sentences = sentences[:top_k] + sentences[-top_k:]
  return new_sentences

mlp_layer = 5
gpt_model = transformer_lens.HookedTransformer.from_pretrained("gpt2").to(device)

# Load in the dataset
ds = load_dataset("JeanKaddour/minipile")

feature_sentences = [[] for i in range(8 * embed_dims)]

data_collection = None
mode = "test"
pbar = tqdm(range(len(ds[mode])), desc="Processing")
count = 0
all_sentences = []
for sample in ds[mode]:
  pbar.update(1)
  count += 1
  # if count > 50:
  #   break

  # Break into sentences and find the activations
  sample_sentences = sample["text"].split(".")
  for x in range(len(sample_sentences)):
    sent = sample_sentences[x]
    all_sentences.append(sent)

    logits, activations = gpt_model.run_with_cache(sent)

    mlp_out = activations.cache_dict[f"blocks.{mlp_layer}.hook_mlp_out"].to(device1) # example MLP output, shape: (1, # samples, # dim)

    y, f, loss, reconstruction_loss = sae(mlp_out, True)
    feature_activations = sae.feature_activations(mlp_out)

    f_by_feature, _ = torch.max(f, dim = 1)
    f_by_feature = f_by_feature.squeeze(0)

    pbar = tqdm(range(f_by_feature.shape[0]), desc=f"Processing {x} / {len(sample_sentences)}")
    f_by_feature_np = f_by_feature.detach().to("cpu").numpy()

    for i in range(f_by_feature.shape[0]):
      pbar.update(1)

      feature_sentences[i].append((f_by_feature_np[i], len(all_sentences) - 1))
    # print(feature_sentences)
  if count % 100 == 0:
    feature_sentences_cp = list(feature_sentences)
    for i in range(len(feature_sentences_cp)):
      feature_sentences_cp[i] = prune_stored_sentences(feature_sentences_cp[i])

    torch.save({
      "activating_sentences": feature_sentences_cp,
      "all_sentences": all_sentences
    }, "activating_sentences1.pt")


for i in range(len(feature_sentences)):
  feature_sentences[i] = prune_stored_sentences(feature_sentences[i])

torch.save({
  "activating_sentences": feature_sentences,
  "all_sentences": all_sentences
}, "activating_sentences1.pt")



