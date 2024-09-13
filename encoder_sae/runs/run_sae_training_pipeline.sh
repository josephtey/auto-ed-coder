#!/bin/bash

python runs/pipeline.py \
    --run_name "final_one" \
    --model_name "joetey/bert-base-uncased-finetuned-set_3" \
    --embedded_data_dir "minipile_20240912_172337" \
    --batch_size 512 \
    --dimensions 768 \
    --sparsity_alpha 1 \
    --lr 0.00001 \
    --num_epochs 1 \
    --sparsity_scale 1 \
    --max_features 20
