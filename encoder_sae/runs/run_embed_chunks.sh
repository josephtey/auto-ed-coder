#!/bin/bash

# python data_preparation/embedding_chunks/embed_chunks.py \
#     --model_name "joetey/bert-base-uncased-finetuned-set_3" \
#     --input_file "data_preparation/chunking/chunked_datasets/JeanKaddour_minipile_20240912_093355.csv" \
#     --folder_name "minipile"

# python data_preparation/embedding_chunks/embed_chunks.py \
#     --model_name "joetey/bert-base-uncased-finetuned-set_3" \
#     --input_file "data_preparation/chunking/chunked_datasets/asap_20240912_101434.csv" \
#     --folder_name "asap"

# python data_preparation/embedding_chunks/embed_chunks.py \
#     --model_name "contra" \
#     --input_file "data_preparation/chunking/chunked_datasets/JeanKaddour_minipile.csv" \
#     --folder_name "contra_minipile"

# python data_preparation/embedding_chunks/embed_chunks.py \
#     --model_name "contra" \
#     --input_file "data_preparation/chunking/chunked_datasets/JeanKaddour_minipile.csv" \
#     --folder_name "contra_minipile" \


python data_preparation/embedding_chunks/embed_chunks.py \
    --model_name "mshenoda/roberta-spam" \
    --input_file "data_preparation/chunking/chunked_datasets/spam_messages_test_20241206_220829_item.csv" \
    --folder_name "spam_test_item" \
    --batch_size 512

# python data_preparation/embedding_chunks/embed_chunks.py \
#     --model_name "nomic-ai/nomic-embed-text-v1" \
#     --input_file "data_preparation/chunking/chunked_datasets/spam_messages_val_20241206_095340.csv" \
#     --folder_name "spam_nomic_valid" \
#     --batch_size 512
