import nltk
import argparse
import csv
import os
from datetime import datetime
from nltk.tokenize import sent_tokenize
from datasets import load_dataset

nltk.download("punkt", quiet=True)


def chunk_dataset(dataset_name, num_sentences):
    ds = load_dataset(dataset_name)
    num_written = 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "data_preparation/chunking/chunked_datasets"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f"{dataset_name.replace('/', '_')}_{timestamp}.csv"
    )

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["sentence"])  # Write header

        for text in ds["train"]["text"]:
            sentences = sent_tokenize(text)
            for sentence in sentences:
                num_written += 1
                writer.writerow([sentence])

                if num_written % 1000 == 0:
                    print(f"Processed {num_written} sentences")

                if num_written >= num_sentences:
                    print(f"Reached the limit of {num_sentences} sentences")
                    break
            if num_written >= num_sentences:
                break

    print(f"Chunking complete. Output saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk a dataset into sentences")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset to chunk")
    parser.add_argument("num_sentences", type=int, help="Number of sentences to chunk")
    args = parser.parse_args()

    chunk_dataset(args.dataset_name, args.num_sentences)
