# Set universal variables
RUN_NAME="spam_messages_test"
MODEL_NAME="mshenoda/roberta-spam"
DATASET="data_preparation/data/spam_messages_test.csv"

# Start by chunking the good ol' dataset
# python data_preparation/chunking/chunk_hf_dataset.py "$DATASET" --chunk_type item --text_column text 

# # Get most recent chunked dataset folder
# chunks_file=$(ls -t data_preparation/chunking/chunked_datasets/*.csv | head -n 1)
chunks_file="data_preparation/chunking/chunked_datasets/spam_messages_train_20241106_110401.csv"

# python data_preparation/embedding_chunks/embed_chunks.py \
#     --model_name "$MODEL_NAME" \
#     --input_file "$chunks_file" \
#     --folder_name "$RUN_NAME"

# # Sleep for 5 seconds to ensure file is written
# sleep 5

# Get most recent embedded chunks folder
# embeddings_folder=$(ls -td data_preparation/embedding_chunks/embedded_chunks/* | head -n 1)
embeddings_folder="data_preparation/embedding_chunks/embedded_chunks/spam_messages_4_20241106_110416"
# python runs/pipeline.py \
#     --run_name "$RUN_NAME" \
#     --model_name "$MODEL_NAME" \
#     --sentences_file "$chunks_file" \
#     --embeddings_file "$embeddings_folder/embeddings.npy" \
#     --batch_size 512 \
#     --dimensions 768 \
#     --sparsity_alpha 1 \
#     --lr 0.00001 \
#     --num_epochs 3 \
#     --sparsity_scale 1

# # Get the most recently created folder in training_sae/saes
latest_sae_folder=$(ls -td training_sae/saes/* | head -n 1)

python feature_extraction/interp_sae.py \
    --sentences_file "$chunks_file" \
    --embeddings_file "$embeddings_folder/embeddings.npy" \
    --sae_base_path "$latest_sae_folder" \
    --features_base_path "feature_extraction/features" \
    --max_features 200 \
    --model "gpt-4o" \
    --feature_registry_path "feature_extraction/features/20241203_014216/feature_registry.npy" \
    --prompt_type "default" \
    --k 50