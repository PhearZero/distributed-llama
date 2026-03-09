#!/bin/bash

# Simple script to run Distributed Llama with Rockchip NPU acceleration

# Configuration
MODEL=${1:-"model.m"}
TOKENIZER=${2:-"tokenizer.t"}
PROMPT=${3:-"Hello, how are you?"}

if [ ! -f "dllama" ]; then
    echo "Error: dllama executable not found. Please run 'make DLLAMA_RKLLM=1' first."
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: Model file '$MODEL' not found."
    exit 1
fi

# Run inference with NPU acceleration enabled (--rkllm 1)
# Add the library directory to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/src/rkllama/lib/

./dllama inference \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --rkllm 1 \
    --prompt "$PROMPT" \
    --steps 128
