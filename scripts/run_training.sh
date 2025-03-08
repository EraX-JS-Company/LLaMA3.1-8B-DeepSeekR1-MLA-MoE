#!/bin/bash
# run_training.sh - Launch script for LLaMA-DeepSeek training
# This script provides different configurations for training on H100 GPUs

set -e  # Exit on error

# Base paths - update these paths for your environment
MODEL_PATH="/path/to/your/llama-deepseek-model"
DATA_PATH="/path/to/your/training/data.json"
OUTPUT_DIR="./output"

# H100 Configuration - update these values based on your GPU count
NUM_GPUS=8
MASTER_PORT=29500

# Default training config
CONTEXT_LENGTH=16384
BATCH_SIZE=6
GRAD_ACCUM=4
LR=1e-5
EPOCHS=3
WARMUP_STEPS=100

# Parse command line arguments
CONFIG=""
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --config)
      CONFIG="$2"
      shift
      shift
      ;;
    --model_path)
      MODEL_PATH="$2"
      shift
      shift
      ;;
    --data_path)
      DATA_PATH="$2"
      shift
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check that paths exist
if [ ! -d "$MODEL_PATH" ]; then
  echo "Error: Model path $MODEL_PATH does not exist"
  exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
  echo "Error: Data path $DATA_PATH does not exist"
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Pre-defined configurations
case $CONFIG in
  # Standard configuration with FSDP (no 8-bit optimizer)
  "fsdp")
    echo "Running with FSDP configuration (standard optimizer)"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_llama_deepseek.py \
      --model_path "$MODEL_PATH" \
      --data_path "$DATA_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --bf16 \
      --context_length $CONTEXT_LENGTH \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --learning_rate $LR \
      --num_epochs $EPOCHS \
      --warmup_steps $WARMUP_STEPS \
      --transformer_engine
    ;;
    
  # Configuration with 8-bit optimizer (uses DDP instead of FSDP)
  "8bit")
    echo "Running with 8-bit optimizer configuration (DDP)"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_llama_deepseek.py \
      --model_path "$MODEL_PATH" \
      --data_path "$DATA_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --bf16 \
      --use_8bit_optimizer \
      --context_length $CONTEXT_LENGTH \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --learning_rate $LR \
      --num_epochs $EPOCHS \
      --warmup_steps $WARMUP_STEPS \
      --transformer_engine
    ;;
    
  # Long context configuration (32K)
  "32k")
    echo "Running with 32K context length configuration"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_llama_deepseek.py \
      --model_path "$MODEL_PATH" \
      --data_path "$DATA_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --bf16 \
      --context_length 32768 \
      --batch_size 4 \
      --gradient_accumulation_steps 6 \
      --learning_rate $LR \
      --num_epochs $EPOCHS \
      --warmup_steps $WARMUP_STEPS \
      --transformer_engine
    ;;
    
  # Extra long context configuration (64K)
  "64k")
    echo "Running with 64K context length configuration"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_llama_deepseek.py \
      --model_path "$MODEL_PATH" \
      --data_path "$DATA_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --bf16 \
      --use_8bit_optimizer \
      --context_length 65536 \
      --batch_size 3 \
      --gradient_accumulation_steps 8 \
      --learning_rate $LR \
      --num_epochs $EPOCHS \
      --warmup_steps 200 \
      --transformer_engine
    ;;
    
  # Ultra long context configuration (128K)
  "128k")
    echo "Running with 128K context length configuration"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_llama_deepseek.py \
      --model_path "$MODEL_PATH" \
      --data_path "$DATA_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --bf16 \
      --use_8bit_optimizer \
      --context_length 131072 \
      --batch_size 2 \
      --gradient_accumulation_steps 12 \
      --learning_rate 5e-6 \
      --num_epochs $EPOCHS \
      --warmup_steps 200 \
      --save_steps 200 \
      --transformer_engine
    ;;

  # Debug configuration - lower batch size, more logging
  "debug")
    echo "Running debug configuration"
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$MASTER_PORT train_llama_deepseek.py \
      --model_path "$MODEL_PATH" \
      --data_path "$DATA_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --bf16 \
      --context_length 2048 \
      --batch_size 1 \
      --gradient_accumulation_steps 1 \
      --num_epochs 1 \
      --debug
    ;;
    
  # Default configuration
  *)
    echo "Running default configuration"
    torchrun --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT train_llama_deepseek.py \
      --model_path "$MODEL_PATH" \
      --data_path "$DATA_PATH" \
      --output_dir "$OUTPUT_DIR" \
      --bf16 \
      --context_length $CONTEXT_LENGTH \
      --batch_size $BATCH_SIZE \
      --gradient_accumulation_steps $GRAD_ACCUM \
      --learning_rate $LR \
      --num_epochs $EPOCHS \
      --warmup_steps $WARMUP_STEPS \
      --transformer_engine
    ;;
esac

echo "Training completed!"
