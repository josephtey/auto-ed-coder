# Run the MiniPile dataset through GPT2 and save the embeddings

from datasets import load_dataset
import transformer_lens
import torch
from tqdm import tqdm
import numpy as np

# Load in mini-GPT2
device = "cuda:0"
device1 = "cuda:1"
mlp_layer = 5
gpt_model = transformer_lens.HookedTransformer.from_pretrained("gpt2").to(device)

# Load in the dataset
ds = load_dataset("JeanKaddour/minˆipile")

data_collection = None
mode = "test"
pbar = tqdm(range(len(ds[mode])), desc="Processing")
count = 0
for sample in ds[mode]:
  pbar.update(1)
  count += 1

  logits, activations = gpt_model.run_with_cache(sample["text"])
  mlp_out = activations.cache_dict[f"blocks.{mlp_layer}.hook_mlp_out"].to(device1) # example MLP output, shape: (1, # samples, # dim)

  if data_collection == None:
    data_collection = mlp_out.to(device1)
  else:
    data_collection = torch.concat([data_collection, mlp_out], dim = 1)

  if count % 1000 == 0:
    np.save(f"mini_pile_{mode}.npy", data_collection.to("cpu").numpy())



