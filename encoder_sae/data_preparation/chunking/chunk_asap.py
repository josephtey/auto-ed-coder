import nltk
import argparse
import csv
import os
from datetime import datetime
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
import pandas as pd

nltk.download("punkt", quiet=True)


def chunk_asap(set_ids):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "data_preparation/chunking/chunked_datasets"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"asap_{timestamp}.csv")

    # Iterate through each set_id
    for set_id in set_ids:
        with open(output_file, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["sentence"])  # Write header

            # Read the CSV file
            df = pd.read_csv(f"../shared/data/asap/asap_{set_id:02d}/df_spell.csv")

            # Iterate through each row in the dataframe
            for essay in df["CorrectedSpellingEssayText"]:
                # Tokenize the essay into sentences
                sentences = sent_tokenize(essay)
                # Add these sentences to our list
                for sentence in sentences:
                    writer.writerow([sentence])

        print(f"Chunking complete. Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk a dataset into sentences")
    parser.add_argument("set_ids", type=int, nargs="+", help="List of set IDs to chunk")
    args = parser.parse_args()

    chunk_asap(args.set_ids)
