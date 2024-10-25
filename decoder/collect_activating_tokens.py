from datasets import load_dataset
import transformer_lens
from tqdm import tqdm
import random

import argparse
import transformer_lens
import torch
import time
from transformers import GPT2Tokenizer


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

from utils import *

from neuron_explainer.activations.activations import ActivationRecord


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

feature_samples = [[] for i in range(8 * embed_dims)]
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

data_collection = None
mode = "test"
pbar = tqdm(range(len(ds[mode])), desc="Processing")
count = 0
all_sentences = []

for sample in ds[mode]:
  pbar.update(1)
  count += 1
  if count > 1000:
    break

  # Get the feature activations for the sample
  model_output = run_sample(gpt_model, sae, sample["text"], device=device1)
  feature_activations, f = model_output["feature_activations"].squeeze(), model_output["f"]

  f_by_feature, _ = torch.max(f, dim = 1)
  f_by_feature = f_by_feature.squeeze(0) # size: # of sample tokens
  f_by_feature_np = f_by_feature.detach().to("cpu").numpy()

  sample_tokens = gpt_model.to_str_tokens(sample["text"])

  # For each feature, create an activation record
  for i in range(feature_activations.shape[1]):
    # Create an ActivationRecord for each feature with a list of tokens and activations for each token
    # Leave the first element out because it's always a <end_of_text> token
    activationRecord = ActivationRecord(tokens=sample_tokens[1:], activations=feature_activations[:, i].detach().to("cpu").numpy()[1:])

    feature_samples[i].append((max(activationRecord.activations), activationRecord))

  # if count % 100 == 0:
  #   feature_samples_cp = list(feature_samples)
  #   for i in range(len(feature_samples_cp)):
  #     feature_samples_cp[i] = prune_stored_sentences(feature_samples_cp[i])

    # torch.save({
    #   "activating_tokens": feature_samples_cp,
    # }, "activating_tokens.pt")


for i in range(len(feature_samples)):
  feature_samples[i] = prune_stored_sentences(feature_samples[i])

torch.save({
  "activating_tokens": feature_samples,
}, "activating_tokens.pt")



