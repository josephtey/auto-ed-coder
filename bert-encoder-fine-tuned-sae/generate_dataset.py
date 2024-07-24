import os
import csv
import numpy as np
import argparse
from tqdm import tqdm
import sys
import torch
from datetime import datetime


def load_embeddings(folder_path, max_num=None):
    embeddings = []
    embedding_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".out")])
    for file in tqdm(embedding_files, desc="Loading embeddings"):
        if max_num is not None and len(embeddings) >= max_num:
            break
        embedding = torch.load(os.path.join(folder_path, file))
        embeddings.append(embedding.numpy())  # Convert to numpy array
    return np.array(embeddings)


def load_sentences(csv_path, num_embeddings, max_num=None):
    sentences = []
    with open(csv_path, "r", encoding="utf-8") as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip header if present
        for row in tqdm(csv_reader, desc="Loading sentences", total=num_embeddings):
            if max_num is not None and len(sentences) >= max_num:
                break
            if len(sentences) >= num_embeddings:
                break
            sentences.append(row[0])  # Assuming the sentence is in the first column
    return np.array(sentences, dtype=object)


def main(embeddings_folder, sentences_file, folder, name, max_num):
    print("Loading embeddings...")
    embeddings = load_embeddings(embeddings_folder, max_num)

    print("Loading sentences...")
    sentences = load_sentences(sentences_file, len(embeddings), max_num)

    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Number of sentences: {len(sentences)}")

    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    embeddings_path = os.path.join(folder, f"{name}_{timestamp}_embeddings.npy")
    sentences_path = os.path.join(folder, f"{name}_{timestamp}_sentences.npy")

    print(f"Saving embeddings to {embeddings_path}...")
    np.save(embeddings_path, embeddings)

    print(f"Saving sentences to {sentences_path}...")
    np.save(sentences_path, sentences)

    print(f"Dataset saved as {embeddings_path} and {sentences_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate separate numpy files for embeddings and sentences"
    )
    parser.add_argument(
        "embeddings_folder", help="Path to the folder containing embedding .out files"
    )
    parser.add_argument("sentences_file", help="Path to the all_sentences.csv file")
    parser.add_argument(
        "--folder",
        default="minipile_dataset",
        help="Folder to save output .npy files",
    )
    parser.add_argument(
        "--name",
        default="dataset",
        help="Name prefix for the output files",
    )
    parser.add_argument(
        "--max_num",
        type=int,
        default=None,
        help="Maximum number of embeddings and sentences to process",
    )

    args = parser.parse_args()

    main(
        args.embeddings_folder,
        args.sentences_file,
        args.folder,
        args.name,
        args.max_num,
    )
