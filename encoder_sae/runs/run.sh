python runs/pipeline.py \
    --run_name "spam_messages_1" \
    --model_name "roberta-spam" \
    --sentences_file "data_preparation/chunking/chunked_datasets/spam_messages_train_20241106_095507.csv" \
    --embeddings_file "data_preparation/embedding_chunks/embedded_chunks/spam_messages_20241106_100045/embeddings.npy" \
    --batch_size 512 \
    --dimensions 768 \
    --sparsity_alpha 1 \
    --lr 0.00001 \
    --num_epochs 8 \
    --sparsity_scale 1 \

# Get the most recently created folder in training_sae/saes
latest_sae_folder=$(ls -td training_sae/saes/* | head -n 1)

python feature_extraction/interp_sae.py \
    --sentences_file "data_preparation/chunking/chunked_datasets/spam_messages_train_20241106_095507.csv" \
    --embeddings_file "data_preparation/embedding_chunks/embedded_chunks/spam_messages_20241106_100045/embeddings.npy" \
    --sae_base_path "$latest_sae_folder" \
    --features_base_path "feature_extraction/features" \
    --max_features 500 \
    --model "gpt-4o-mini" 