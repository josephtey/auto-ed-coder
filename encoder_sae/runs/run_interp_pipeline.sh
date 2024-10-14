#!/bin/bash

python feature_extraction/interp_sae.py \
    --sentences_file "data_preparation/embedding_chunks/embedded_chunks/contra_minipile_20241013_110604/sentences_checkpoint.npy" \
    --embeddings_file "data_preparation/embedding_chunks/embedded_chunks/contra_minipile_20241013_110604/embeddings_checkpoint.npy" \
    --sae_base_path "training_sae/saes/contra_1_20241013_173503" \
    --features_base_path "feature_extraction/features" \
    --max_features 100 \
    --model "gpt-4o-mini"
