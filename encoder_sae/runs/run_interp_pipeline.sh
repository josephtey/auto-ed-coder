#!/bin/bash

python feature_extraction/interp_sae.py \
    --sentences_file "data_preparation/chunking/chunked_datasets/JeanKaddour_minipile_20241104_184439.csv" \
    --embeddings_file "data_preparation/embedding_chunks/embedded_chunks/minipile_20241104_184603/embeddings.npy" \
    --sae_base_path "training_sae/saes/cls_token_l5_20241118_011028" \
    --features_base_path "feature_extraction/features" \
    --max_features 50 \
    --model "gpt-4o-mini" \
    --feature_registry_path "feature_extraction/features/feature_registry.npy"