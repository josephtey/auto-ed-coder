import nltk

nltk.download("punkt")
from nltk.tokenize import sent_tokenize
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch

bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased")

batch_size = 10
df_all_sentences = pd.read_csv("all_sentences.csv")
sentences = df_all_sentences["sentence"].tolist()

idx = 0
for i in range(24560, len(sentences), batch_size):
    batch = sentences[i : i + batch_size]
    try:
        inputs = bert_tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True
        )
        outputs = bert_model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach()

        for embedding in batch_embeddings:
            with open(f"embeddings/{idx}.out", "wb") as f:
                torch.save(embedding, f)

            idx += 1

    except Exception as e:
        print(f"Error: {e}")
        continue

    print(f"Processed {i + len(batch)} sentences")
