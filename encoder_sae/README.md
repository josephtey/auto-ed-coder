```
embedding_models/
│
├── fine_tune_model.ipynb    # Fine-tune the model (takes a param for model selection)
└── models/                  # Directory for all the fine-tuned models

data_preparation/
│
├── chunking/
│   ├── chunk_asap.py         # Script to chunk ASAP format data
│   ├── chunk_hf_dataset.py   # Script to chunk Hugging Face datasets
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
