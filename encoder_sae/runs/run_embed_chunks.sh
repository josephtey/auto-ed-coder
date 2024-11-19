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
    --model_name "bert-base-uncased" \
    --input_file "data_preparation/chunking/chunked_datasets/JeanKaddour_minipile_20241104_184439.csv" \
    --folder_name "minipile"
