#!/bin/bash

python feature_extraction/interp_sae.py \
    --sentences_file "data_preparation/embedding_chunks/embedded_chunks/asap_20240912_172932/sentences.npy" \
    --embeddings_file "data_preparation/embedding_chunks/embedded_chunks/asap_20240912_172932/embeddings.npy" \
    --sae_base_path "training_sae/saes/final_one_20240913_082034" \
    --features_base_path "feature_extraction/features" \
    --max_features 20 \
    --model "gpt-4o-mini"
