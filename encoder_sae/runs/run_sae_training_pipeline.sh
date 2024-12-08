#!/bin/bash

# python runs/pipeline.py \
#     --run_name "cls_token_l5" \
#     --model_name "bert-base-uncased" \
#     --sentences_file "data_preparation/chunking/chunked_datasets/JeanKaddour_minipile_20241104_184439.csv" \
#     --embeddings_file "data_preparation/embedding_chunks/embedded_chunks/minipile_20241105_040126/embeddings.npy" \
#     --batch_size 512 \
#     --dimensions 768 \
#     --sparsity_alpha 1 \
#     --lr 0.0001 \
#     --num_epochs 100 \
#     --sparsity_scale 1 \
#     --max_features 20

# python runs/pipeline.py \
#     --run_name "contra_3" \
#     --model_name "contra-large" \
#     --sentences_file "data_preparation/chunking/chunked_datasets/JeanKaddour_minipile.csv" \
#     --embeddings_file "data_preparation/embedding_chunks/embedded_chunks/contra_minipile_20241014_002936/embeddings.npy" \
#     --batch_size 512 \
#     --dimensions 1024 \
#     --sparsity_alpha 1 \
#     --lr 0.00001 \
#     --num_epochs 1 \
#     --sparsity_scale 1 \
#     --max_features 20

python runs/pipeline.py \
    --run_name "spam_messages_1" \
    --model_name "roberta-spam" \
    --sentences_file "data_preparation/chunking/chunked_datasets/spam_messages_train_20241106_095507.csv" \
    --embeddings_file "data_preparation/embedding_chunks/embedded_chunks/spam_messages_20241106_095541/embeddings.npy" \
    --batch_size 512 \
    --dimensions 768 \
    --sparsity_alpha 1 \
    --lr 0.00001 \
    --num_epochs 1 \
    --sparsity_scale 1 \
