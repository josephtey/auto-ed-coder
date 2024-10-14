#!/bin/bash

# python runs/pipeline.py \
#     --run_name "final_one" \
#     --model_name "joetey/bert-base-uncased-finetuned-set_3" \
#     --embedded_data_dir "minipile_20240912_172337" \
#     --batch_size 512 \
#     --dimensions 768 \
#     --sparsity_alpha 1 \
#     --lr 0.00001 \
#     --num_epochs 1 \
#     --sparsity_scale 1 \
#     --max_features 20

python runs/pipeline.py \
    --run_name "contra_1" \
    --model_name "contra-base" \
    --sentences_file "data_preparation/chunking/chunked_datasets/JeanKaddour_minipile.csv" \
    --embeddings_file "data_preparation/embedding_chunks/embedded_chunks/embedded_chunks/contra_minipile_20241013_110604/embeddings_checkpoint.npy" \
    --batch_size 512 \
    --dimensions 768 \
    --sparsity_alpha 1 \
    --lr 0.00001 \
    --num_epochs 1 \
    --sparsity_scale 1 \
    --max_features 100 \
