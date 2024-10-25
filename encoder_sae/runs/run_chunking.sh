#!/bin/bash

python data_preparation/chunking/chunk_hf_dataset.py JeanKaddour/minipile 1000
python data_preparation/chunking/chunk_asap.py 3
