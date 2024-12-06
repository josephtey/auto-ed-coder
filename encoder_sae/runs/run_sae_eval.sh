#!/bin/bash

# Directory containing the SAE model
# SAE_DIR="training_sae/saes/spam_messages_roberta_20241206_094425"
SAE_DIR="training_sae/saes/spam_messages_roberta_tied_weights_20241206_111202"


# Test data paths
TEST_EMBEDDINGS="data_preparation/embedding_chunks/embedded_chunks/spam_valid_20241206_095537/embeddings.npy"
# TEST_EMBEDDINGS="data_preparation/embedding_chunks/embedded_chunks/spam_nomic_valid_20241206_095945/embeddings.npy"
TEST_SENTENCES="data_preparation/chunking/chunked_datasets/spam_messages_val_20241206_095340.csv"

# Batch size for evaluation
BATCH_SIZE=512

# Run the evaluation script
python evaluate_sae/evaluate.py \
    --sae-dir "$SAE_DIR" \
    --test-embeddings "$TEST_EMBEDDINGS" \
    --test-sentences "$TEST_SENTENCES" \
    --batch-size "$BATCH_SIZE"
