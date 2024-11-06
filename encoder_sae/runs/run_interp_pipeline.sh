#!/bin/bash

python feature_extraction/interp_sae.py \
    --sentences_file "data_preparation/chunking/chunked_datasets/JeanKaddour_minipile.csv" \
    --embeddings_file "data_preparation/embedding_chunks/embedded_chunks/contra_minipile_20241013_201035/embeddings.npy" \
    --sae_base_path "training_sae/saes/contra_2_20241013_220300" \
    --features_base_path "feature_extraction/features" \
    --max_features 500 \
    --model "gpt-4o-mini" \
    --feature_registry_path "feature_extraction/features/feature_registry.npy"