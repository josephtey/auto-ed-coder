import numpy as np
from utils.ai import OpenAIClient
import os
import json
from pprint import pprint
from datetime import datetime
import pickle
from utils.sae import SparseAutoencoder, SparseAutoencoderConfig
import json
from dotenv import load_dotenv
from utils.features import Feature, FeatureSample
import argparse
from utils.models import MiniPileDataset
import tqdm

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")


def main(args):

    # load the dataset
    file_name = args.dataset_filename
    with open(file_name, "rb") as f:
        mini_pile_dataset = pickle.load(f)

    # Load the configuration from the JSON file
    config_path = os.path.join(args.sae_base_path, "config.json")
    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    # Load the pre-trained model from the pickle file
    sae_config = SparseAutoencoderConfig(
        d_model=config["dimensions"],
        d_sparse=8 * config["dimensions"],
        sparsity_alpha=config["sparsity_alpha"],
    )
    model = SparseAutoencoder(sae_config)
    model_path = os.path.join(args.sae_base_path, "sae.pkl")
    with open(model_path, "rb") as f:
        model_state_dict = pickle.load(f)
        model.load_state_dict(model_state_dict)

    # make folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = os.path.join(args.features_base_path, f"sae_features_{timestamp}")
    os.makedirs(folder_name, exist_ok=True)

    ai = OpenAIClient(openai_api_key)

    n = len(mini_pile_dataset)
    feature_registry = np.zeros((config["dimensions"] * 8, n))

    for i in tqdm.tqdm(range(n), desc="Creating feature registry"):
        embedding = mini_pile_dataset.embeddings[i]
        feature_activations = model.forward(embedding)[1]
        feature_registry[:, i] = feature_activations.detach().numpy()

    for index, feature in enumerate(
        tqdm.tqdm(feature_registry, desc="Writing features")
    ):
        feature_samples = [
            FeatureSample(text=mini_pile_dataset.sentences[i], act=value)
            for i, value in enumerate(feature)
        ]
        feature_samples.sort(key=lambda x: x.act, reverse=True)

        high_act_samples = feature_samples[:50]
        low_act_samples = feature_samples[-50:]

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
            index=index,
            label=label,
            attributes=attributes,
            reasoning=reasoning,
            confidence=abs(high_act_score - low_act_score),
            density=(np.count_nonzero(feature) / len(feature)),
            high_act_samples=high_act_samples,
            low_act_samples=low_act_samples,
        )

        # write this feature
        with open(os.path.join(folder_name, f"feature_{index}.json"), "w") as json_file:
            json.dump(labelled_feature.dict(), json_file, indent=4)

        # print processed feature
        print(f"Processed feature {index}: {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpret Sparse Autoencoder Features"
    )
    parser.add_argument(
        "--dataset_filename", type=str, required=True, help="Path to the dataset file"
    )
    parser.add_argument(
        "--sae_base_path",
        type=str,
        required=True,
        help="Base path for SAE config and model files",
    )
    parser.add_argument(
        "--features_base_path",
        type=str,
        required=True,
        help="Base path to save the features",
    )

    args = parser.parse_args()
    main(args)
