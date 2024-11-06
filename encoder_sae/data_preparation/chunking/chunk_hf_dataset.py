import nltk
import argparse
import csv
import os
import pandas as pd
from datetime import datetime
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from tqdm import tqdm

nltk.download("punkt", quiet=True)


def chunk_dataset(input_source, num_chunks=None, chunk_type="sentence", text_column=None):
    if input_source.endswith('.csv'):
        df = pd.read_csv(input_source)
        if text_column is None:
            raise ValueError("text_column must be specified for CSV input")
        texts = df[text_column].tolist()
    else:
        ds = load_dataset(input_source)
        texts = ds["train"]["text"]

    num_written = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "data_preparation/chunking/chunked_datasets"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"{os.path.basename(input_source).split('.')[0]}_{timestamp}.csv"
    )

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sentence"])  # Write header

        total_items = len(texts)
        for text in tqdm(texts, total=total_items, desc="Processing chunks"):
            if chunk_type == "sentence":
                chunks = sent_tokenize(text)
            else:  # per item
                chunks = [text]
            
            for chunk in chunks:
                num_written += 1
                writer.writerow([chunk])

                if num_chunks is not None and num_written >= num_chunks:
                    print(f"Reached the limit of {num_chunks} chunks")
                    break
            if num_chunks is not None and num_written >= num_chunks:
                break

    print(f"Chunking complete. Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk a dataset or CSV file")
    parser.add_argument("input_source", type=str, help="Name of the dataset or path to CSV file to chunk")
    parser.add_argument("--num_chunks", type=int, help="Number of chunks to process (optional)")
    parser.add_argument("--chunk_type", type=str, choices=["sentence", "item"], default="sentence",
                        help="Type of chunking: 'sentence' or 'item' (default: sentence)")
    parser.add_argument("--text_column", type=str, help="Column name for text in CSV file (required for CSV input)")
    args = parser.parse_args()

    chunk_dataset(args.input_source, args.num_chunks, args.chunk_type, args.text_column)
