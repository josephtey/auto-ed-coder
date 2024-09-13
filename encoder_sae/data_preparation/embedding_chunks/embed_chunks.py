import argparse
import os
import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import json


def embed_chunks(model_name, input_file, folder_name):
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)

    batch_size = 40

    df_all_sentences = pd.read_csv(input_file)
    sentences = df_all_sentences["sentence"].tolist()

    all_embeddings = []
    all_sentences = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    embedded_chunks_dir = os.path.join(
        "data_preparation/embedding_chunks/embedded_chunks",
        f"{folder_name}_{timestamp}",
    )
    os.makedirs(embedded_chunks_dir, exist_ok=True)

    checkpoint_interval = 20  # Save checkpoint every 20 batches

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        try:
            inputs = bert_tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            )

            with torch.no_grad():  # Disable gradient calculation
                outputs = bert_model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach()

            all_embeddings.extend(batch_embeddings.cpu().numpy())
            all_sentences.extend(batch)

            # Clear cache regularly
            del inputs
            del outputs
            del batch_embeddings
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error: {e}")
            continue

        print(f"Processed {i + len(batch)} sentences")

        # Save checkpoint
        if (i // batch_size + 1) % checkpoint_interval == 0:
            np.save(
                os.path.join(embedded_chunks_dir, "embeddings_checkpoint.npy"),
                np.array(all_embeddings),
            )
            np.save(
                os.path.join(embedded_chunks_dir, "sentences_checkpoint.npy"),
                np.array(all_sentences),
            )
            print(f"Checkpoint saved at batch {i // batch_size + 1}")

    # Save all embeddings and sentences to .npy files
    np.save(
        os.path.join(embedded_chunks_dir, "embeddings.npy"), np.array(all_embeddings)
    )
    np.save(os.path.join(embedded_chunks_dir, "sentences.npy"), np.array(all_sentences))

    # Save config of the dataset
    config = {
        "model_name": model_name,
        "input_file": input_file,
    }
    with open(os.path.join(embedded_chunks_dir, "config.json"), "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed sentences using an embedding model."
    )
    parser.add_argument("--model_name", type=str, help="Name of the embedding model.")
    parser.add_argument(
        "--input_file", type=str, help="Path to the input CSV file containing sentences"
    )
    parser.add_argument(
        "--folder_name",
        type=str,
        help="Name of the folder to save the output embedded chunks",
    )

    args = parser.parse_args()
    embed_chunks(args.model_name, args.input_file, args.folder_name)
