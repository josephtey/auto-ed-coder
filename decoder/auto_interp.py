

import torch
import sys
sys.path.append("../")
from shared.ai import OpenAIClient
from shared.features import Feature, FeatureSample
import os
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Write our feature to disk
def write_labelled_feature_to_file(labelled_feature):
  with open(
    os.path.join("./decoder/features", f"feature_{labelled_feature.index}.json"), "w"
  ) as json_file:
    json.dump(labelled_feature.dict(), json_file, indent=4)

activating_samples = torch.load("./decoder/activating_sentences.pt")
all_sentences = activating_samples["all_sentences"]

ai = OpenAIClient(openai_api_key)

cnt = 0
for feature_activation in activating_samples["activating_sentences"]:
  cnt += 1
  print(cnt)

  feature_samples = [
    FeatureSample(text=all_sentences[f_sample_idx], act=f_act)
    for f_act, f_sample_idx in feature_activation
  ]
  feature_samples.sort(key=lambda x: x.act)

  high_act_samples = feature_samples[len(feature_samples) // 2:]
  low_act_samples = feature_samples[:len(feature_samples) // 2]
  try:
    interpetation = ai.get_interpretation(high_act_samples, low_act_samples)
    label = interpetation["label"]
    reasoning = interpetation["reasoning"]
    attributes = interpetation["attributes"]

    high_act_score = ai.score_interpretation(high_act_samples, attributes)[
        "percent"
    ]
    low_act_score = ai.score_interpretation(low_act_samples, attributes)[
        "percent"
    ]
  except Exception as e:
    print(f"Skipping feature due to error: {e}")
    continue

  labelled_feature = Feature(
    index=cnt,
    label=label,
    attributes=attributes,
    reasoning=reasoning,
    confidence=abs(high_act_score - low_act_score),
    density=0,
    high_act_samples=high_act_samples,
    low_act_samples=low_act_samples,
  )
  write_labelled_feature_to_file(labelled_feature)
