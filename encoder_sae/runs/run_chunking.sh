#!/bin/bash

python data_preparation/chunking/chunk_hf_dataset.py data_preparation/data/spam_messages_val.csv --chunk_type sentence --text_column text
# python data_preparation/chunking/chunk_asap.py 3
