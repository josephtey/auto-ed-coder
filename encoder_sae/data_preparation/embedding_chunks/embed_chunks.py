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

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


class BottleneckT5Autoencoder:
    def __init__(self, model_path: str, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, model_max_length=512, use_fast=False
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, text: str) -> torch.FloatTensor:
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        decoder_inputs = self.tokenizer("", return_tensors="pt").to(self.device)
        return self.model(
            **inputs,
            decoder_input_ids=decoder_inputs["input_ids"],
            encode_only=True,
        )[0]

    @torch.no_grad()
    def generate_from_latent(
        self, latent: torch.FloatTensor, max_length=512, temperature=1.0
    ) -> str:
        dummy_text = "."
        dummy = self.embed(dummy_text)
        perturb_vector = latent - dummy
        self.model.perturb_vector = perturb_vector
        input_ids = (
            self.tokenizer(dummy_text, return_tensors="pt").to(self.device).input_ids
        )
        output = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            num_return_sequences=1,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


def embed_chunks(
    model_name,
    input_file,
    folder_name,
    resume_from_checkpoint=False,
    checkpoint_dir=None,
):

    if model_name == "contra":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        autoencoder = BottleneckT5Autoencoder(
            model_path="thesephist/contra-bottleneck-t5-large-wikipedia", device=device
        )
    else:
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)

    batch_size = 40

    df_all_sentences = pd.read_csv(input_file)
    sentences = df_all_sentences["sentence"].tolist()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if checkpoint_dir:
        embedded_chunks_dir = checkpoint_dir
    else:
        embedded_chunks_dir = os.path.join(
            "data_preparation/embedding_chunks/embedded_chunks",
            f"{folder_name}_{timestamp}",
        )
    os.makedirs(embedded_chunks_dir, exist_ok=True)

    checkpoint_interval = 2500  # Save checkpoint every 2500 batches

    if resume_from_checkpoint:
        # Load the latest checkpoint
        checkpoint_files = [
            f for f in os.listdir(embedded_chunks_dir) if f.endswith("_checkpoint.npy")
        ]
        if checkpoint_files:
            all_embeddings = list(
                np.load(os.path.join(embedded_chunks_dir, "embeddings_checkpoint.npy"))
            )
            start_index = len(all_embeddings)
            print(
                f"Resuming from checkpoint in {embedded_chunks_dir}. Starting from index {start_index}"
            )
        else:
            print(
                f"No checkpoint found in {embedded_chunks_dir}. Starting from the beginning."
            )
            all_embeddings = []
            start_index = 0
    else:
        all_embeddings = []
        start_index = 0

    for i in range(start_index, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        try:
            if model_name == "contra":
                for s in batch:
                    embedding = autoencoder.embed(s).cpu().numpy()
                    all_embeddings.append(embedding)
            else:
                inputs = bert_tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=True
                )
                with torch.no_grad():  # Disable gradient calculation
                    outputs = bert_model(**inputs)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].detach()

                all_embeddings.extend(batch_embeddings.cpu().numpy())

                del inputs
                del outputs
                del batch_embeddings

            # clear cache
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
            print(f"Checkpoint saved at batch {i // batch_size + 1}")

    # Save all embeddings and sentences to .npy files
    np.save(
        os.path.join(embedded_chunks_dir, "embeddings.npy"), np.array(all_embeddings)
    )

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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory containing the checkpoint to resume from",
    )

    args = parser.parse_args()
    embed_chunks(
        args.model_name,
        args.input_file,
        args.folder_name,
        args.resume,
        args.checkpoint_dir,
    )
