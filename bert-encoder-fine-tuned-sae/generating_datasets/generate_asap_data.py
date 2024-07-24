import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import argparse

# Download the punkt tokenizer if not already downloaded
nltk.download("punkt", quiet=True)


def process_asap_data(set_ids):
    # Initialize an empty list to store all sentences
    all_sentences = []

    # Iterate through each set_id
    for set_id in set_ids:
        # Read the CSV file
        df = pd.read_csv(f"data/asap_{set_id:02d}/df_spell.csv")

        # Iterate through each row in the dataframe
        for essay in df["CorrectedSpellingEssayText"]:
            # Tokenize the essay into sentences
            sentences = sent_tokenize(essay)
            # Add these sentences to our list
            all_sentences.extend(sentences)

    # Load pre-trained BERT model and tokenizer
    bert_tokenizer = AutoTokenizer.from_pretrained(
        "joetey/bert-base-uncased-finetuned-set_3"
    )
    bert_model = AutoModel.from_pretrained("joetey/bert-base-uncased-finetuned-set_3")

    # Set batch size
    batch_size = 40

    # Initialize list to store embeddings
    all_embeddings = []

    # Process sentences in batches
    for i in tqdm(
        range(0, len(all_sentences), batch_size), desc="Processing sentences"
    ):
        batch = all_sentences[i : i + batch_size]
        try:
            # Tokenize and encode the batch
            inputs = bert_tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True
            )

            # Generate embeddings
            with torch.no_grad():
                outputs = bert_model(**inputs)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach()

            # Add batch embeddings to the list
            all_embeddings.extend(batch_embeddings.numpy())

            # Clear cache
            del inputs
            del outputs
            del batch_embeddings
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing batch: {e}")
            continue

    # Convert lists to numpy arrays
    sentences_array = np.array(all_sentences, dtype=object)
    embeddings_array = np.array(all_embeddings)

    # Save arrays to files
    # Create a string of set_ids joined by underscores
    set_ids_str = "_".join(map(str, args.set_ids))

    # Save arrays to files with set_ids in the filename
    np.save(f"pickled_data/asap_sentences_{set_ids_str}.npy", sentences_array)
    np.save(f"pickled_data/asap_embeddings_{set_ids_str}.npy", embeddings_array)

    print(f"Total sentences processed: {len(sentences_array)}")
    print(f"Sentences saved to asap_sentences.npy")
    print(f"Embeddings saved to asap_embeddings.npy")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ASAP dataset")
    parser.add_argument(
        "--set_ids",
        nargs="+",
        type=int,
        required=True,
        help="List of ASAP set IDs to process",
    )
    args = parser.parse_args()

    process_asap_data(args.set_ids)
