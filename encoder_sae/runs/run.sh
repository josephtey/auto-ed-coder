# Set universal variables
RUN_NAME="spam_messages_4"
MODEL_NAME="mshenoda/roberta-spam"
DATASET="data_preparation/data/spam_messages_train.csv"

# Start by chunking the good ol' dataset
python data_preparation/chunking/chunk_hf_dataset.py "$DATASET" --chunk_type sentence --text_column text 

# Get most recent chunked dataset folder
latest_chunks_file=$(ls -t data_preparation/chunking/chunked_datasets/*.csv | head -n 1)

python data_preparation/embedding_chunks/embed_chunks.py \
    --model_name "$MODEL_NAME" \
    --input_file "$latest_chunks_file" \
    --folder_name "$RUN_NAME"

# Get most recent embedded chunks folder
latest_embeddings_folder=$(ls -td data_preparation/embedding_chunks/embedded_chunks/* | head -n 1)

python runs/pipeline.py \
    --run_name "$RUN_NAME" \
    --model_name "$MODEL_NAME" \
    --sentences_file "$latest_chunks_file" \
    --embeddings_file "$latest_embeddings_folder/embeddings.npy" \
    --batch_size 512 \
    --dimensions 768 \
    --sparsity_alpha 1 \
    --lr 0.00001 \
    --num_epochs 3 \
    --sparsity_scale 1

# Get the most recently created folder in training_sae/saes
latest_sae_folder=$(ls -td training_sae/saes/* | head -n 1)

python feature_extraction/interp_sae.py \
    --sentences_file "data_preparation/chunking/chunked_datasets/spam_messages_train_20241106_095507.csv" \
    --embeddings_file "data_preparation/embedding_chunks/embedded_chunks/${RUN_NAME}_20241106_100045/embeddings.npy" \
    --sae_base_path "$latest_sae_folder" \
    --features_base_path "feature_extraction/features" \
    --max_features 200 \
    --model "gpt-4o-mini"