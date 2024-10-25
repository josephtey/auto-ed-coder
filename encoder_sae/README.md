```
embedding_models/
│
├── fine_tune_model.ipynb    # Fine-tune ENCODER models for embedding
└── models/                  # Directory for all the fine-tuned ENCODER models

data_preparation/
│
├── chunking/
│   ├── chunk_asap.py         # Script to chunk ASAP data (short anaswer scoring dataset from Kaggle)
│   ├── chunk_hf_dataset.py   # Script to chunk Hugging Face datasets (we use minipile right now)
│   └── chunked_datasets/     # Datasets already chunked (includes checkpoints)
│
└── embedding_chunks/
    ├── embed_chunks.py       # Script to embed chunks
    └── embedded_chunks/      # Directory for all embedded chunks

training_sae/
│
├── train_sae.py              # Script to train SAEs
└── saes/                     # Directory for all trained SAEs

feature_extraction/
│
├── interp.sae.py             # Script for interpreting SAEs and feature extraction
└── features/                 # Directory for all extracted features

runs/
│
├── run_chunking.sh               # Shell script to run chunking process
├── run_embed_chunks.sh           # Shell script to run embedding process
├── run_sae_training_pipeline.sh  # Shell script to run SAE training pipeline
└── run_interp_pipeline.sh        # Shell script to run interpretation pipeline

analysis/
└── analysis.ipynb            # Jupyter notebook for analyzing results
```

#### How to run

0. `fine_tune_bert.ipynb`: Script to fine-tune BERT models. Right now, the SAEs are trained from BERT encodings, in this case, specifically for auto-grading.
1. `run_chunking.sh`: Chunks dataset into sentences and then stores them in `chunked_datasets`. Could be interesting to see how different chunk sizes will impact things (to be added).
2. `run_embed_chunks.sh`: Embeds these chunks using the fine-tuned model from Step 0, on sentence level.
3. `run_sae_training_pipeline`: Trains an SAE on a set of provided embeddings. SAE is like an 'eye' to semantically understand a series of numbers just a little better.
4. `run_interp_pipeline`: Extracts features of the SAE based on some 'test' dataset that we use!
