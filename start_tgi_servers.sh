#!/bin/bash

# Share cache directory
volume=/fastdata/hfcache/transformers/
# HF token
token=$(cat ${HOME}/.config/huggingface/token)

# StarCoder: 8192, GPUs 0,1
#port=8192
#model=bigcode/starcoder
#docker run -d -e HUGGING_FACE_HUB_TOKEN=$token --gpus '"device=0,1"' --shm-size 1g \
#    -p ${port}:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest \
#    --model-id $model --trust-remote-code --dtype bfloat16 --sharded true --num-shard 2 \
#    --max-total-tokens 8192 --max-input-length 8000 --max-batch-prefill-tokens 8000

# Code Llama: 8193, GPUs 2,3
port=8192
model=codellama/CodeLlama-13b-hf
docker run -e HUGGING_FACE_HUB_TOKEN=$token --gpus '"device=0,3"' --shm-size 1g \
    -p ${port}:80 -v $volume:/data ghcr.io/huggingface/text-generation-inference:latest \
    --model-id $model --trust-remote-code --dtype bfloat16 --sharded true --num-shard 2 \
    --max-total-tokens 8192 --max-input-length 8000 --max-batch-prefill-tokens 8000
