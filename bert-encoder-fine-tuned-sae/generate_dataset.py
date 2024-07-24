import os
import csv
import pickle
import numpy as np
import argparse
from tqdm import tqdm
import sys

sys.path.append("../")
from shared.models import MiniPileDataset
import torch


def load_embeddings(folder_path):
    embeddings = []
    embedding_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".out")])
    for file in tqdm(embedding_files, desc="Loading embeddings"):
        embedding = torch.load(os.path.join(folder_path, file))
        embeddings.append(embedding)
    return torch.stack(embeddings)


def load_sentences(csv_path, num_embeddings):
    sentences = []
    with open(csv_path, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip header if present
        for row in tqdm(csv_reader, desc="Loading sentences", total=num_embeddings):
            if len(sentences) >= num_embeddings:
                break
            sentences.append(row[0])  # Assuming the sentence is in the first column
    return sentences


def main(embeddings_folder, sentences_file, output_file):
    print("Loading embeddings...")
    embeddings = load_embeddings(embeddings_folder)

    print("Loading sentences...")
    sentences = load_sentences(sentences_file, len(embeddings))

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Number of sentences: {len(sentences)}")

    print("Creating MiniPileDataset...")
    dataset = MiniPileDataset(sentences, embeddings)

    print(f"Saving dataset to {output_file}...")
    with open(output_file, "wb") as f:
        pickle.dump(dataset, f)

    print(f"Dataset saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MiniPileDataset from embeddings and sentences"
    )
    parser.add_argument(
        "embeddings_folder", help="Path to the folder containing embedding .out files"
    )
    parser.add_argument("sentences_file", help="Path to the all_sentences.csv file")
    parser.add_argument(
        "--output",
        default="minipile_dataset.pkl",
        help="Path to save the output pickle file",
    )

    args = parser.parse_args()

    main(args.embeddings_folder, args.sentences_file, args.output)
