import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize
import csv
from datasets import load_dataset

ds = load_dataset("JeanKaddour/minipile")

with open("all_sentences.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["sentence"])  # Write header

    for text in ds["train"]["text"]:
        sentences = sent_tokenize(text)
        for sentence in sentences:
            writer.writerow([sentence])