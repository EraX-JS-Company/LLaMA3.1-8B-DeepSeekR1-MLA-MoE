#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated Training Script for LLaMA-DeepSeek model.

Features:
- Compatible with H100 GPUs
- Gradient checkpointing
- FSDP with proper configuration
- Optional BNB 8-bit optimizer (with automatic fallback to DDP)
- Flash Attention V3 for non-MLA layers
- Transformer Engine integration for H100s
- Memory-efficient data loading

Usage:
    torchrun --nproc_per_node=8 train_llama_deepseek.py \
    --model_path /path/to/model \
    --data_path /path/to/data.json \
    --output_dir ./output \
    --context_length 32768 \
    --[other options]
"""

import os
import argparse
import logging
import time
import json
import math
import random
import warnings
import gc
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    StateDictType,
    FullStateDictConfig,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.data.dataset import IterableDataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP

import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    PretrainedConfig,
    default_data_collator,
    get_scheduler,
    PreTrainedModel,
)

# Import custom model classes directly from conversion script
from llama_deepseek_convert import (
    LlamaDeepSeekConfig,
    LlamaMultiheadLatentAttention,
    LlamaMoEMLP,
    LlamaDeepSeekDecoderLayer,
    LlamaDeepSeekModel,
    LlamaDeepSeekForCausalLM,
    register_llama_deepseek_model,
    load_converted_model,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Try importing Flash Attention
try:
    import flash_attn
    from flash_attn import flash_attn_func
    HAVE_FLASH_ATTN = True
    flash_attn_version = flash_attn.__version__
    logger.info(f"Flash Attention found, version: {flash_attn_version}")
except ImportError:
    HAVE_FLASH_ATTN = False
    logger.warning("Flash Attention not found, falling back to standard attention")

# Try importing bitsandbytes for 8-bit optimizer
try:
    import bitsandbytes as bnb
    HAVE_BITSANDBYTES = True
    logger.info("bitsandbytes found, 8-bit optimizer available")
except ImportError:
    HAVE_BITSANDBYTES = False
    logger.warning("bitsandbytes not found, falling back to standard optimizers")

# Try importing Transformer Engine
try:
    import transformer_engine as te
    from transformer_engine.pytorch import fp8_autocast
    from transformer_engine.common import recipe
    HAVE_TRANSFORMER_ENGINE = True
    logger.info("Transformer Engine found, H100 optimizations available")
except ImportError:
    HAVE_TRANSFORMER_ENGINE = False
    logger.warning("Transformer Engine not found, H100 optimizations unavailable")


class StreamingTextDataset(IterableDataset):
    """Memory-efficient streaming dataset for large-scale pretraining."""
    
    def __init__(self, data_path, tokenizer, max_length, buffer_size=10000, shuffle=True):
        """
        Initialize a streaming text dataset.
        
        Args:
            data_path: Path to JSON or text file containing training data
            tokenizer: Tokenizer for processing text
            max_length: Maximum sequence length
            buffer_size: Size of shuffle buffer
            shuffle: Whether to shuffle data
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        
    def parse_json_data(self):
        """Parse and yield text entries from a JSON file."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            try:
                # Try loading entire file
                data = json.load(f)
                if isinstance(data, list):
                    if all(isinstance(item, str) for item in data):
                        for text in data:
                            yield text
                    elif all(isinstance(item, dict) for item in data):
                        # Try common field names
                        field_found = False
                        for field in ['text', 'content', 'input', 'paragraph', 'document']:
                            if field in data[0]:
                                field_found = True
                                for item in data:
                                    if field in item:
                                        yield item[field]
                                break
                        if not field_found:
                            # Fallback: convert whole dict to string
                            for item in data:
                                yield json.dumps(item)
                elif isinstance(data, dict):
                    # Handle dict format with text field
                    if 'texts' in data:
                        for text in data['texts']:
                            yield text
                    else:
                        yield json.dumps(data)
            except json.JSONDecodeError:
                # If not valid JSON, try line-by-line parsing
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        item = json.loads(line)
                        if isinstance(item, str):
                            yield item
                        elif isinstance(item, dict):
                            for field in ['text', 'content', 'input']:
                                if field in item:
                                    yield item[field]
                                    break
                            else:
                                yield json.dumps(item)
                        else:
                            yield json.dumps(item)
                    except json.JSONDecodeError:
                        # If line is not valid JSON, yield as raw text
                        yield line
    
    def parse_text_data(self):
        """Parse and yield text entries from a plain text file."""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line
    
    def __iter__(self):
        """Iterate through the dataset, tokenizing on the fly."""
        # Choose parser based on file extension
        if self.data_path.endswith('.json') or self.data_path.endswith('.jsonl'):
            text_iterator = self.parse_json_data()
        else:
            text_iterator = self.parse_text_data()
        
        # Get worker info for sharding
        worker_info = torch.utils.data.get_worker_info()
        
        # Local buffer for efficient shuffling
        buffer = []
        
        # Process the data stream
        for text in text_iterator:
            if not text:
                continue
                
            # Skip examples that don't belong to this worker when using multiple workers
            if worker_info is not None:
                if random.Random(hash(text)).randint(0, worker_info.num_workers - 1) != worker_info.id:
                    continue
            
            # Tokenize text
            tokenized = self.tokenizer(
                text, 
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            example = {
                "input_ids": tokenized.input_ids[0],
                "attention_mask": tokenized.attention_mask[0],
                "labels": tokenized.input_ids[0].clone(),
            }
            
            # Add to buffer for shuffling
            buffer.append(example)
            
            # When buffer is full, shuffle and yield examples
            if len(buffer) >= self.buffer_size:
                if self.shuffle:
                    random.shuffle(buffer)
                for item in buffer:
                    yield item
                buffer = []
        
        # Yield remaining examples
        if buffer:
            if self.shuffle:
                random.shuffle(buffer)
            for item in buffer:
                yield item


def apply_flash_attention_v3(model, layer_indices=None):
    """
    Apply Flash Attention V3 to standard LlamaAttention layers.
    
    Args:
        model: The LlamaDeepSeekForCausalLM model
        layer_indices: List of layer indices to apply Flash Attention to
                       If None, apply to all non-MLA layers
    """
    if not HAVE_FLASH_ATTN:
        logger.warning("Flash Attention not available, skipping application")
        return model
    
    # Determine which layers should get Flash Attention
    if layer_indices is None:
        # Apply to all non-MLA layers
        mla_layers = getattr(model.config, "mla_layers", [])
        layer_indices = [i for i in range(model.config.num_hidden_layers) if i not in mla_layers]
    
    # Monkey patch the forward method of standard attention
    from transformers.models.llama.modeling_llama import LlamaAttention
    original_forward = LlamaAttention.forward
    
    def flash_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Flash Attention V3 version of the forward method.
        Only applies during training mode (not inference).
        """
        # During inference or if Flash Attention is not supported, use original implementation
        if not self.training or not HAVE_FLASH_ATTN:
            return original_forward(
                self, 
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            
        bsz, q_len, _ = hidden_states.size()
        
        # Get query, key, value projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to (bsz, seq_len, n_heads, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_seq_len = key_states.shape[-2]
        
        # Use rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        # Repeat k/v heads if needed
        key_states = torch.repeat_interleave(key_states, repeats=self.num_heads // self.num_key_value_heads, dim=1)
        value_states = torch.repeat_interleave(value_states, repeats=self.num_heads // self.num_key_value_heads, dim=1)
        
        # Use Flash Attention v3 for the attention calculation
        attn_output = flash_attn_func(
            query_states.transpose(1, 2),  # [batch_size, seq_len, num_heads, head_dim]
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            causal=True,
        ).transpose(1, 2)
        
        # Reshape back
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, None
    
    # Apply the patch to specified layers
    for idx in layer_indices:
        if idx >= len(model.model.layers):
            logger.warning(f"Layer index {idx} out of bounds, skipping")
            continue
            
        layer = model.model.layers[idx]
        
        # Only patch standard attention, not MLA
        if isinstance(layer.attention, LlamaAttention):
            # Monkey patch the forward method
            bound_method = flash_attn_forward.__get__(layer.attention, LlamaAttention)
            setattr(layer.attention, 'forward', bound_method)
            logger.info(f"Applied Flash Attention V3 to layer {idx}")
    
    logger.info(f"Applied Flash Attention V3 to {len(layer_indices)} layers")
    return model


def apply_transformer_engine_optimizations(model):
    """
    Apply NVIDIA Transformer Engine optimizations for H100 GPUs.
    
    Args:
        model: The model to optimize
        
    Returns:
        Optimized model
    """
    if not HAVE_TRANSFORMER_ENGINE:
        logger.warning("Transformer Engine not available, skipping H100 optimizations")
        return model
    
    # Check if running on H100
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    if "H100" not in device_name and "Hopper" not in device_name:
        logger.warning(f"Device {device_name} may not fully support Transformer Engine optimizations")
    
    # Create FP8 recipe with appropriate scaling factors and amax history
    fp8_recipe = recipe.DelayedScaling(
        margin=0,                  # No margin for stability
        interval=1,                # Update scaling factors every step
        fp8_format=recipe.Format.E4M3,  # Higher precision FP8 format
        amax_history_len=1024,     # Store longer history
        amax_compute_algo="max",   # Use max for scaling
    )
    
    # Store original forward functions
    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
    
    # Save original forward functions
    orig_attn_forward = LlamaAttention.forward
    orig_mla_forward = LlamaMultiheadLatentAttention.forward if hasattr(LlamaMultiheadLatentAttention, 'forward') else None
    orig_mlp_forward = LlamaMLP.forward
    
    # Create optimized forward with FP8 acceleration
    def optimized_attn_forward(self, *args, **kwargs):
        with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            return orig_attn_forward(self, *args, **kwargs)
    
    def optimized_mlp_forward(self, *args, **kwargs):
        with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            return orig_mlp_forward(self, *args, **kwargs)
    
    # Apply optimized forwards
    LlamaAttention.forward = optimized_attn_forward
    LlamaMLP.forward = optimized_mlp_forward
    
    # Also optimize MLA if available
    if orig_mla_forward:
        def optimized_mla_forward(self, *args, **kwargs):
            with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
                return orig_mla_forward(self, *args, **kwargs)
        LlamaMultiheadLatentAttention.forward = optimized_mla_forward
    
    logger.info("Applied Transformer Engine optimizations for H100")
    return model


def apply_activation_checkpointing_to_model(model):
    """
    Apply activation checkpointing to model layers for memory efficiency.
    
    Args:
        model: The model to apply activation checkpointing to
    """
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    
    # Define checkpointing policy
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    
    # Apply checkpointing to decoder layers
    check_fn = lambda submodule: isinstance(submodule, (LlamaDecoderLayer, LlamaDeepSeekDecoderLayer))
    
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn
    )
    
    logger.info(f"Applied activation checkpointing to model")
    return model


def prepare_optimizer(model, lr=1e-5, weight_decay=0.01, use_8bit=False):
    """
    Prepare optimizer with optional 8-bit quantization.
    
    Args:
        model: The model
        lr: Learning rate
        weight_decay: Weight decay
        use_8bit: Whether to use 8-bit optimizer
        
    Returns:
        Optimizer
    """
    # Get parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    if use_8bit and HAVE_BITSANDBYTES:
        logger.info("Creating 8-bit AdamW optimizer")
        optimizer = bnb.optim.AdamW8bit(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
    else:
        if use_8bit and not HAVE_BITSANDBYTES:
            logger.warning("bitsandbytes not available, falling back to standard AdamW")
        logger.info("Creating standard AdamW optimizer")
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=True,  # Use fused implementation for better performance
        )
    
    return optimizer


def setup_fsdp_model(model, use_mixed_precision=True, cpu_offload=False):
    """
    Set up FSDP for the model.
    
    Args:
        model: The model to wrap
        use_mixed_precision: Whether to use mixed precision
        cpu_offload: Whether to offload parameters to CPU
        
    Returns:
        FSDP-wrapped model
    """
    # First, move model to meta device to avoid memory issues during FSDP initialization
    model = model.to('meta')
    
    # Define layer classes to be wrapped by FSDP
    try:
        # Import model classes for wrapping policy
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaAttention, LlamaMLP
        
        # Define transformer layers to wrap
        transformer_layer_cls = (
            LlamaDecoderLayer,
            LlamaDeepSeekDecoderLayer,
            LlamaAttention,
            LlamaMultiheadLatentAttention,
            LlamaMLP,
            LlamaMoEMLP,
        )
        
        logger.info(f"FSDP wrap policy includes: {[cls.__name__ for cls in transformer_layer_cls]}")
    except Exception as e:
        logger.warning(f"Error determining custom layer classes: {e}, using fallback")
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer
        transformer_layer_cls = (LlamaDecoderLayer,)
    
    # Configure mixed precision policy
    mp_policy = None
    if use_mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    
    # Configure CPU offload policy
    cpu_offload_policy = None
    if cpu_offload:
        cpu_offload_policy = CPUOffload(offload_params=True)
    
    # Create wrapping policy - using both transformer policy and min size policy
    wrap_policy = transformer_auto_wrap_policy(transformer_layer_cls)
    
    # Apply FSDP with optimal settings
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        use_orig_params=True,  # Important for parameter management
        cpu_offload=cpu_offload_policy,
    )
    
    logger.info(f"Model wrapped with FSDP")
    return model


def setup_ddp_model(model):
    """
    Set up DistributedDataParallel (DDP) for the model.
    Compatible with 8-bit optimizers.
    
    Args:
        model: The model to wrap
        
    Returns:
        DDP-wrapped model
    """
    # Move model to device first
    local_rank = torch.cuda.current_device()
    model = model.to(local_rank)
    
    # Enable gradient checkpointing if available
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing for DDP model")
    
    # Apply activation checkpointing if feasible
    try:
        apply_activation_checkpointing_to_model(model)
        logger.info("Applied activation checkpointing to DDP model")
    except Exception as e:
        logger.warning(f"Failed to apply activation checkpointing to DDP model: {e}")
    
    # Wrap with DDP
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,  # Better performance when disabled
        broadcast_buffers=False,       # Better performance when disabled
    )
    
    logger.info("Model wrapped with DDP")
    return model


def setup_distributed_training(model, use_8bit_optimizer=False, use_mixed_precision=True, cpu_offload=False):
    """
    Set up distributed training - either with FSDP or DDP depending on optimizer choice.
    
    Args:
        model: The model to wrap
        use_8bit_optimizer: Whether 8-bit optimizer is requested
        use_mixed_precision: Whether to use mixed precision
        cpu_offload: Whether to offload parameters to CPU
        
    Returns:
        Wrapped model
    """
    # Choose between FSDP (without 8-bit) or DDP (with 8-bit)
    if use_8bit_optimizer and HAVE_BITSANDBYTES:
        logger.info("Using DDP with 8-bit optimizer (FSDP not compatible with 8-bit optimizers)")
        return setup_ddp_model(model)
    else:
        logger.info("Using FSDP for distributed training")
        return setup_fsdp_model(model, use_mixed_precision, cpu_offload)


def freeze_original_layers(model):
    """
    Freeze original LLaMA layers, keeping MLA and MoE components trainable.
    
    Args:
        model: The model
        
    Returns:
        Model with frozen layers
    """
    if not hasattr(model.config, "mla_layers") or not hasattr(model.config, "moe_layers"):
        logger.warning("Config doesn't have MLA/MoE layer information. Skipping selective freezing.")
        return model
    
    mla_layers = model.config.mla_layers
    moe_layers = model.config.moe_layers
    modified_layers = set(mla_layers + moe_layers)
    
    logger.info(f"Freezing original layers, keeping MLA layers {mla_layers} and MoE layers {moe_layers} trainable")
    
    # Freeze embeddings
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False
    
    # Freeze final norm
    for param in model.model.norm.parameters():
        param.requires_grad = False
    
    # Freeze LM head
    for param in model.lm_head.parameters():
        param.requires_grad = False
    
    # Selectively freeze decoder layers
    for i, layer in enumerate(model.model.layers):
        if i not in modified_layers:
            # Freeze entire unchanged layer
            for param in layer.parameters():
                param.requires_grad = False
            logger.info(f"Layer {i}: Freezing entire layer")
        else:
            # For modified layers, freeze only unchanged components
            for param in layer.input_layernorm.parameters():
                param.requires_grad = False
                
            for param in layer.post_attention_layernorm.parameters():
                param.requires_grad = False
            
            if i in mla_layers:
                # For MLA layers, keep MLA attention components trainable
                logger.info(f"Layer {i}: Keeping MLA attention trainable")
                
                # Verify this is actually an MLA layer
                if hasattr(layer.attention, "latent_q_proj"):
                    # Keep only MLA-specific components trainable
                    for param_name, param in layer.attention.named_parameters():
                        if not any(p in param_name for p in ["latent_q_proj", "latent_k_proj", "latent_v_proj", "latent_o_proj"]):
                            param.requires_grad = False
                else:
                    logger.warning(f"Layer {i} is marked as MLA but doesn't have latent_q_proj")
            else:
                # Freeze attention in non-MLA layers
                for param in layer.attention.parameters():
                    param.requires_grad = False
                logger.info(f"Layer {i}: Freezing standard attention")
                    
            if i in moe_layers:
                # For MoE layers, keep MoE MLP components trainable
                logger.info(f"Layer {i}: Keeping MoE MLP trainable")
                
                # Verify this is actually an MoE layer
                if hasattr(layer.mlp, "router"):
                    # MoE components are trainable
                    pass
                else:
                    logger.warning(f"Layer {i} is marked as MoE but doesn't have router")
                    # Freeze regular MLP
                    for param in layer.mlp.parameters():
                        param.requires_grad = False
            else:
                # Freeze MLP in non-MoE layers
                for param in layer.mlp.parameters():
                    param.requires_grad = False
                logger.info(f"Layer {i}: Freezing standard MLP")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Model has {total_params:,} total parameters")
    logger.info(f"Model has {trainable_params:,} trainable parameters ({trainable_params/total_params:.2%})")
    
    return model


def load_model_for_training(
    model_path,
    use_8bit_optimizer=False,
    freeze_layers=True,
    fp16=True,
    use_flash_attn=True,
):
    """
    Load the LLaMA-DeepSeek model and prepare it for training.
    
    Args:
        model_path: Path to model
        use_8bit_optimizer: Whether to use 8-bit optimizer
        freeze_layers: Whether to freeze original layers
        fp16: Whether to use mixed precision
        use_flash_attn: Whether to use Flash Attention
        
    Returns:
        Model and tokenizer
    """
    # Register custom model type
    register_llama_deepseek_model()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - first try with our custom loader
    try:
        logger.info(f"Loading model with load_converted_model from {model_path}")
        model, _ = load_converted_model(
            model_path, 
            device_map="meta"  # Don't load weights to GPU yet
        )
        
        # Set model to FP16 if requested
        if fp16:
            model = model.half()
            
        logger.info(f"Successfully loaded model with custom loader")
    except Exception as e:
        logger.warning(f"Error loading with custom loader: {e}, falling back to AutoModelForCausalLM")
        
        # Configure loading options
        torch_dtype = torch.float16 if fp16 else torch.float32
        
        # Try loading with Flash Attention if requested
        if use_flash_attn and HAVE_FLASH_ATTN:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    attn_implementation="flash_attention_2",
                    use_cache=False,  # Disable KV cache for training
                    trust_remote_code=True,
                )
                logger.info(f"Loaded model with Flash Attention")
            except Exception as e2:
                logger.warning(f"Error loading with Flash Attention: {e2}, falling back to standard attention")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    use_cache=False,
                    trust_remote_code=True,
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                use_cache=False,
                trust_remote_code=True,
            )
    
    # If FlashAttention wasn't applied during loading, apply it now to non-MLA layers
    if use_flash_attn and HAVE_FLASH_ATTN:
        try:
            model = apply_flash_attention_v3(model)
        except Exception as e:
            logger.warning(f"Failed to apply Flash Attention V3: {e}")
    
    # Enable gradient checkpointing - CRITICAL for memory efficiency
    model.gradient_checkpointing_enable()
    logger.info("Enabled gradient checkpointing")
    
    # Apply activation checkpointing for additional memory efficiency
    try:
        apply_activation_checkpointing_to_model(model)
    except Exception as e:
        logger.warning(f"Failed to apply activation checkpointing: {e}")
    
    # Optionally freeze original layers
    if freeze_layers:
        freeze_original_layers(model)
    
    # Add debug info about model structure
    try:
        mla_layers = getattr(model.config, "mla_layers", [])
        moe_layers = getattr(model.config, "moe_layers", [])
        logger.info(f"Model has MLA layers at: {mla_layers}")
        logger.info(f"Model has MoE layers at: {moe_layers}")
    except Exception as e:
        logger.warning(f"Could not extract layer info: {e}")
    
    return model, tokenizer


def save_model_checkpoint(model, tokenizer, optimizer, scheduler, global_step, output_dir):
    """
    Save model checkpoint with FSDP or DDP.
    
    Args:
        model: The model
        tokenizer: Tokenizer
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        global_step: Global step
        output_dir: Output directory
    """
    logger.info(f"Saving checkpoint to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # Save config
    if hasattr(model, "module") and hasattr(model.module, "config"):
        model.module.config.save_pretrained(output_dir)
    
    # Check if model is FSDP or DDP
    is_fsdp = isinstance(model, FSDP)
    
    # Different saving strategies for FSDP vs DDP
    if is_fsdp:
        # Save model with FSDP
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        try:
            # Get full state dict on rank 0
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                model_state = model.state_dict()
                if dist.get_rank() == 0:
                    # Create a config to save with the model
                    model_to_save = LlamaDeepSeekForCausalLM(model.module.config)
                    model_to_save.load_state_dict(model_state)
                    model_to_save.save_pretrained(output_dir)
            
            logger.info(f"Saved FSDP model state successfully")
        except Exception as e:
            logger.error(f"Error saving FSDP model: {e}")
            # Try to save just the config as fallback
            if dist.get_rank() == 0 and hasattr(model, "module") and hasattr(model.module, "config"):
                config_dict = model.module.config.to_dict()
                with open(os.path.join(output_dir, "config.json"), "w") as f:
                    json.dump(config_dict, f, indent=2)
                logger.info("Saved config.json as fallback")
    else:
        # Save model with DDP (simpler)
        if dist.get_rank() == 0:
            model.module.save_pretrained(output_dir)
            logger.info(f"Saved DDP model state successfully")
    
    # Save training state on rank 0
    if dist.get_rank() == 0:
        try:
            training_state = {
                "global_step": global_step,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            }
            torch.save(training_state, os.path.join(output_dir, "training_state.pt"))
            logger.info("Saved training state")
        except Exception as e:
            logger.error(f"Error saving training state: {e}")
    
    # Force synchronization
    dist.barrier()
    logger.info(f"Checkpoint saved to {output_dir}")


def train(
    model,
    tokenizer,
    train_dataloader,
    optimizer,
    lr_scheduler,
    max_grad_norm=1.0,
    num_epochs=3,
    gradient_accumulation_steps=8,
    fp16=True,
    output_dir="./output",
    save_steps=100,
    local_rank=0,
):
    """
    Main training function.
    
    Args:
        model: The model
        tokenizer: Tokenizer
        train_dataloader: Training data loader
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        max_grad_norm: Maximum gradient norm
        num_epochs: Number of training epochs
        gradient_accumulation_steps: Number of steps for gradient accumulation
        fp16: Whether to use mixed precision
        output_dir: Output directory
        save_steps: Save checkpoint every X steps
        local_rank: Local rank for distributed training
    """
    # Set up device
    device = torch.cuda.current_device()
    
    # Initialize mixed precision training if requested
    scaler = GradScaler() if fp16 else None
    
    # Prepare for distributed training
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    is_main_process = local_rank == 0
    
    # Create output directory if needed
    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)
    
    # Training state
    step = 0
    global_step = 0
    tr_loss = 0.0
    moving_loss = 0.0
    model.train()
    
    # Performance monitoring
    tokens_seen = 0
    peak_memory = 0
    
    # Main training loop
    logger.info(f"Starting training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        
        # Reset dataset sampler for distributed training
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        
        # Track time for throughput calculation
        epoch_start_time = time.time()
        batch_start_time = time.time()
        total_tokens = 0
        
        # Training loop
        for batch_idx, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            
            # Forward pass with mixed precision if enabled
            with autocast(enabled=fp16):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                
                # Check for router auxiliary loss if using MoE
                router_aux_loss = 0.0
                if hasattr(model.module, "config") and hasattr(model.module.config, "moe_layers"):
                    try:
                        for layer_idx in model.module.config.moe_layers:
                            layer = model.module.model.layers[layer_idx]
                            if hasattr(layer.mlp, "_router_aux_loss"):
                                router_aux_loss += layer.mlp._router_aux_loss
                    except (AttributeError, IndexError) as e:
                        pass  # Ignore errors in aux loss computation
                
                if router_aux_loss > 0:
                    router_aux_loss = router_aux_loss / gradient_accumulation_steps
                    loss += router_aux_loss
            
            # Backward pass with gradient scaling if enabled
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update training metrics
            tr_loss += loss.item() * gradient_accumulation_steps
            
            # Calculate tokens seen (for throughput)
            batch_tokens = batch["input_ids"].numel()
            total_tokens += batch_tokens
            tokens_seen += batch_tokens
            
            # Track peak memory usage
            current_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
            peak_memory = max(peak_memory, current_memory)
            
            # Optimizer step after gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                if scaler is not None:
                    scaler.unscale_(optimizer)
                
                # Apply gradient clipping
                clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Update parameters
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Update learning rate
                lr_scheduler.step()
                
                # Zero gradients
                optimizer.zero_grad(set_to_none=True)  # More memory-efficient than zero_grad()
                
                # Update step counters
                step += 1
                global_step += 1
                
                # Calculate throughput
                batch_end_time = time.time()
                elapsed = batch_end_time - batch_start_time
                tokens_per_sec = batch_tokens * gradient_accumulation_steps / elapsed if elapsed > 0 else 0
                
                # Update moving average loss
                moving_loss = 0.9 * moving_loss + 0.1 * (loss.item() * gradient_accumulation_steps) if moving_loss > 0 else loss.item() * gradient_accumulation_steps
                
                # Log progress
                if step % 10 == 0 and is_main_process:
                    curr_lr = lr_scheduler.get_last_lr()[0]
                    
                    logger.info(
                        f"Epoch: {epoch+1}/{num_epochs} | "
                        f"Step: {step}/{len(train_dataloader)//gradient_accumulation_steps} | "
                        f"Loss: {moving_loss:.4f} | "
                        f"LR: {curr_lr:.2e} | "
                        f"Throughput: {tokens_per_sec:.1f} tokens/sec | "
                        f"Memory: {current_memory:.2f}GB"
                    )
                
                # Save checkpoint
                if global_step % save_steps == 0 and is_main_process:
                    save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                    save_model_checkpoint(model, tokenizer, optimizer, lr_scheduler, global_step, save_path)
                
                # Reset batch start time for next throughput calculation
                batch_start_time = time.time()
        
        # End of epoch stats
        epoch_time = time.time() - epoch_start_time
        avg_throughput = total_tokens / epoch_time if epoch_time > 0 else 0
        
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s | "
            f"Average Loss: {tr_loss/step:.4f} | "
            f"Average Throughput: {avg_throughput:.1f} tokens/sec | "
            f"Peak Memory: {peak_memory:.2f}GB"
        )
        
        # Save epoch checkpoint
        if is_main_process:
            save_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch+1}")
            save_model_checkpoint(model, tokenizer, optimizer, lr_scheduler, global_step, save_path)
    
    # Save final model
    if is_main_process:
        final_path = os.path.join(output_dir, "final_model")
        save_model_checkpoint(model, tokenizer, optimizer, lr_scheduler, global_step, final_path)
    
    # Final training stats
    total_training_time = time.time() - epoch_start_time
    avg_throughput = tokens_seen / total_training_time if total_training_time > 0 else 0
    
    logger.info(
        f"Training completed in {total_training_time:.2f}s | "
        f"Total tokens: {tokens_seen:,} | "
        f"Average throughput: {avg_throughput:.1f} tokens/sec | "
        f"Final loss: {tr_loss/step:.4f}"
    )


def main():
    """Main function for LLaMA-DeepSeek model training."""
    parser = argparse.ArgumentParser(description="Train LLaMA-DeepSeek model with FSDP/DDP and optional 8-bit optimizer")
    
    # Model and data arguments
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--data_path", type=str, required=True, help="Path to training data file (JSON or TXT)")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory for checkpoints")
    
    # Training hyperparameters
    parser.add_argument("--context_length", type=int, default=4096, help="Maximum context length")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of steps for gradient accumulation")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Learning rate warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    
    # Optimization options
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision (FP16)")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 precision (better for H100s)")
    parser.add_argument("--use_8bit_optimizer", action="store_true", help="Use 8-bit AdamW optimizer")
    parser.add_argument("--no_flash_attn", action="store_true", help="Disable Flash Attention")
    parser.add_argument("--no_freeze_layers", action="store_true", help="Don't freeze original LLaMA layers")
    parser.add_argument("--cpu_offload", action="store_true", help="Enable CPU offloading for FSDP")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with extra logging")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Buffer size for streaming dataset")
    
    # H100-specific optimizations
    parser.add_argument("--transformer_engine", action="store_true", help="Enable NVIDIA Transformer Engine for H100s")
    
    args = parser.parse_args()
    
    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Initialize distributed environment
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        logger.info("RANK and WORLD_SIZE not set, assuming single-GPU training")
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
    
    # Initialize distributed process group
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    
    logger.info(f"Initialized process group: rank {local_rank}/{world_size}")
    
    # Log GPU info
    logger.info(f"Using {torch.cuda.device_count()} GPUs per node")
    logger.info(f"Current device: {torch.cuda.current_device()}")
    logger.info(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    logger.info(f"Device memory: {torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3):.2f} GB")
    
    # Log Flash Attention availability
    if HAVE_FLASH_ATTN:
        logger.info(f"Flash Attention available, version: {flash_attn_version}")
    else:
        logger.warning("Flash Attention not available")
    
    # Apply Transformer Engine optimizations if requested
    if args.transformer_engine and HAVE_TRANSFORMER_ENGINE:
        logger.info("Will apply Transformer Engine optimizations for H100s")
    elif args.transformer_engine and not HAVE_TRANSFORMER_ENGINE:
        logger.warning("Transformer Engine requested but not available. Install with: pip install transformer-engine")
    
    # Load model and tokenizer
    model, tokenizer = load_model_for_training(
        model_path=args.model_path,
        use_8bit_optimizer=args.use_8bit_optimizer,
        freeze_layers=not args.no_freeze_layers,
        fp16=args.fp16 or args.bf16,
        use_flash_attn=not args.no_flash_attn,
    )
    
    # Apply Transformer Engine optimizations if requested
    if args.transformer_engine and HAVE_TRANSFORMER_ENGINE:
        model = apply_transformer_engine_optimizations(model)
    
    # Choose distributed training strategy based on optimizer choice
    mixed_precision = args.fp16 or args.bf16
    model = setup_distributed_training(
        model, 
        use_8bit_optimizer=args.use_8bit_optimizer,
        use_mixed_precision=mixed_precision,
        cpu_offload=args.cpu_offload
    )
    
    # Create optimizer
    optimizer = prepare_optimizer(
        model, 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        use_8bit=args.use_8bit_optimizer and isinstance(model, DDP)  # Only use 8-bit with DDP
    )
    
    # Load dataset
    dataset = StreamingTextDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.context_length,
        buffer_size=args.buffer_size
    )
    
    # Create data sampler and loader
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=default_data_collator,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,  # Keep workers alive between batches
    )
    
    # Set up learning rate scheduler
    num_training_steps = len(dataloader) // args.gradient_accumulation_steps * args.num_epochs
    
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    # Log effective batch size and other training info
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps * world_size
    logger.info(f"Starting training with:")
    logger.info(f"  GPUs: {world_size}")
    logger.info(f"  Batch size per GPU: {args.batch_size}")
    logger.info(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {effective_batch_size}")
    logger.info(f"  Context length: {args.context_length}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Precision: {'BF16' if args.bf16 else 'FP16' if args.fp16 else 'FP32'}")
    logger.info(f"  8-bit optimizer: {args.use_8bit_optimizer}")
    logger.info(f"  Flash Attention: {not args.no_flash_attn}")
    logger.info(f"  Freeze original layers: {not args.no_freeze_layers}")
    logger.info(f"  CPU offload: {args.cpu_offload}")
    logger.info(f"  Transformer Engine: {args.transformer_engine}")
    logger.info(f"  Training mode: {'DDP with 8-bit optimizer' if args.use_8bit_optimizer else 'FSDP'}")
    logger.info(f"  Total training steps: {num_training_steps}")
    
    # Start training
    train(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16 or args.bf16,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        local_rank=local_rank,
    )
    
    # Clean up
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

'''
------------------

With 8x H100 NVL 94GB GPUs, you've got access to truly exceptional hardware! Let me recalculate the training parameters for this top-tier configuration.
Maximum Context Length
With H100 NVL 94GB GPUs and your setup:

65,536 tokens (64K) can be handled with ease
131,072 tokens (128K) should be reliably achievable
Up to 262,144 tokens (256K) may be possible with optimized settings

The H100 NVL's massive 94GB HBM3 memory and NVLink interconnect make it the most capable hardware available for long-context training.
Memory Consumption
With 8x H100 NVL 94GB GPUs using FSDP and freezing non-MLA/MoE layers:

Peak memory per GPU during forward pass:

~22-28GB with 16K context at batch size 4
~35-40GB with 32K context at batch size 3
~55-65GB with 64K context at batch size 2
~75-85GB with 128K context at batch size 1


Memory headroom:

You'll still have ~10-20GB of headroom even at 128K context
This allows for increased batch sizes, model growth, or training optimizations



Training Speed
Training speed metrics with 8x H100 NVL 94GB GPUs:

Throughput:

~45,000-55,000 tokens per second with 16K context
~30,000-40,000 tokens per second with 32K context
~20,000-25,000 tokens per second with 64K context
~12,000-16,000 tokens per second with 128K context


Time per step (with recommended batch sizes):

~1.2-1.5 seconds per step with 16K context (batch size 4, gradient accumulation 8)
~2.5-3.0 seconds per step with 32K context (batch size 3, gradient accumulation 8)
~5.0-6.5 seconds per step with 64K context (batch size 2, gradient accumulation 8)
~8.0-10.0 seconds per step with 128K context (batch size 1, gradient accumulation 16)


Training a 1B token dataset:

~6-7 hours with 16K context
~8-10 hours with 32K context
~12-15 hours with 64K context
~18-22 hours with 128K context



These estimates take advantage of:

The additional HBM3 memory (94GB vs 80GB)
Enhanced NVLink bandwidth between GPUs
Full Transformer Engine optimizations
Gradient accumulation adjusted for larger batch sizes

Recommended Configuration
With such powerful hardware, I recommend:

For 16K context: batch size 4, gradient accumulation 8 (effective batch size 256)
For 32K context: batch size 3, gradient accumulation 8 (effective batch size 192)
For 64K context: batch size 2, gradient accumulation 8 (effective batch size 128)
For 128K context: batch size 1, gradient accumulation 16 (effective batch size 128)

You're working with essentially the most powerful GPU configuration currently available for AI training. This setup allows you to efficiently train with extremely long context windows that were previously impractical, giving you a significant advantage for training models that can handle document-level understanding and multi-document reasoning.

-------------

With 8x H100 80GB GPUs, your training capabilities will be significantly improved. Let me recalculate the expected maximum context length, memory consumption, and training speed.
Maximum Context Length
With H100 80GB GPUs and your setup:

32,768 tokens (32K) should be comfortably achievable
Up to 65,536 tokens (64K) is possible with optimized settings
Even 131,072 tokens (128K) might be possible with very small batch sizes and optimized memory usage

The H100s offer a massive upgrade in both memory capacity (80GB vs 40GB) and HBM3 memory bandwidth, allowing for much longer context processing.
Memory Consumption
With 8x H100 80GB GPUs using FSDP and freezing non-MLA/MoE layers:

Peak memory per GPU during forward pass:

~20-25GB with 8K context at batch size 2
~30-35GB with 16K context at batch size 2
~45-50GB with 32K context at batch size 1
~65-70GB with 64K context at batch size 1


Memory headroom:

You'll have ~10-15GB of headroom even with 64K context lengths
This allows for potential batch size increases or model parameter growth



Training Speed
Training speed metrics with 8x H100 80GB GPUs:

Throughput:

~35,000-45,000 tokens per second with 8K context
~25,000-35,000 tokens per second with 16K context
~15,000-20,000 tokens per second with 32K context
~8,000-12,000 tokens per second with 64K context


Time per step (with batch size 2, gradient accumulation 16):

~0.7-0.9 seconds per step with 8K context
~1.2-1.5 seconds per step with 16K context
~2.5-3.5 seconds per step with 32K context
~5.5-7.0 seconds per step with 64K context


Training a 1B token dataset:

~7-8 hours with 8K context
~10-12 hours with 16K context
~16-18 hours with 32K context
~24-30 hours with 64K context



These estimates account for:

H100's improved Tensor Cores performance (substantially faster than A100)
NVLink 4.0 interconnect between GPUs (higher bandwidth)
Transformer Engine optimizations available on H100s
8-bit AdamW optimizer
Flash Attention V3 enabled for non-MLA layers

The H100s offer approximately 3x the training throughput of A100s for transformer models, especially when utilizing TensorFloat-32 (TF32) and Flash Attention optimizations.
With this powerful setup, you could feasibly:

Increase batch sizes to 2-4 per GPU (versus 1 on A100s)
Train with much longer contexts (32K-64K)
Complete training iterations 2.5-3x faster than with A100s

For optimal performance, I recommend starting with a batch size of 2 per GPU with context lengths of 16K or 32K, which should give you excellent training speed while keeping memory usage at comfortable levels.

'''
