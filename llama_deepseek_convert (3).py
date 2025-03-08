#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLaMA-3.1 8B to DeepSeek R1 MLA+MoE Conversion Utility

This script converts a LLaMA-3.1 8B model to incorporate DeepSeek R1's 
Multi-head Latent Attention (MLA) and Mixture of Experts (MoE) architecture.

It selectively replaces certain layers with MLA and MoE while preserving
as much of the original model weights as possible to maintain model quality.
"""

import os
import math
import random
import warnings
from typing import Optional, Tuple, Union, List, Dict, Any
import time
import logging
import json
import argparse
from dataclasses import dataclass, field, asdict
from copy import deepcopy
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import torch.distributed as dist

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm, 
    LlamaRotaryEmbedding, 
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    LlamaForCausalLM,
    apply_rotary_pos_emb
)

import triton
import flash_attn
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash_attn.bert_padding import pad_input, unpad_input

import numpy as np
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ============================================================================
# DeepSeek R1 MLA+MoE Configuration
# ============================================================================

@dataclass
class LlamaDeepSeekConfig(LlamaConfig):
    """
    Configuration class for LLaMA model with DeepSeek R1 MLA and MoE extensions.
    Combines LLaMA-3.1 architecture with DeepSeek's improvements.
    """
    model_type: str = "llama_deepseek"
    
    # LLaMA-3.1 base params
    hidden_size: int = 4096
    intermediate_size: int = 11008
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8  # GQA in LLaMA-3.1
    hidden_act: str = "silu"
    max_position_embeddings: int = 131072  # Context length 128K as requested
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = None
    bos_token_id: int = 1
    eos_token_id: int = 2
    attention_dropout: int = None
    
    # DeepSeek R1 MLA-specific params
    use_mla: bool = True
    mla_layers: List[int] = field(default_factory=lambda: [8, 16, 24, 30])  # Layers to convert to MLA
    num_latent_heads: int = 16  # Number of latent heads in MLA
    latent_size: int = 1024  # Latent size in MLA
    
    # DeepSeek R1 MoE-specific params
    use_moe: bool = True
    moe_layers: List[int] = field(default_factory=lambda: [10, 18, 26, 31])  # Layers to convert to MoE
    num_experts: int = 8  # Total number of experts as requested
    num_experts_per_tok: int = 2  # Active experts per token as requested
    expert_router_type: str = "top"  # Router type (top-k selection)
    router_aux_loss_coef: float = 0.01
    
    # Extra params from LLaMA-3.1
    attention_bias: bool = False
    tie_word_embeddings: bool = False
    rope_theta: float = 10000.0
    rope_scaling: Optional[Dict[str, Any]] = None
    
    def __init__(self, **kwargs):
        """Initialize with custom handling for vocab_size parameter."""
        # Handle vocab_size separately since it's required but not a class attribute
        vocab_size = kwargs.pop('vocab_size', 128256)  # LLaMA-3.1 8B vocab size
        super().__init__(vocab_size=vocab_size, **kwargs)

# ============================================================================
# DeepSeek R1 Multi-head Latent Attention (MLA) Implementation
# ============================================================================

class LlamaMultiheadLatentAttention(nn.Module):
    """
    Multi-head Latent Attention (MLA) implementation for LLaMA, based on DeepSeek R1 architecture.
    MLA introduces a latent space for the queries, reducing computation while maintaining performance.
    """
    def __init__(self, config: LlamaDeepSeekConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        
        # MLA specific parameters
        self.num_latent_heads = config.num_latent_heads
        self.latent_size = config.latent_size
        
        # Validate parameters
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
            
        # Projection matrices
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        # MLA specific projections
        self.latent_q_proj = nn.Linear(self.hidden_size, self.latent_size, bias=False)
        self.latent_k_proj = nn.Linear(self.latent_size, self.num_latent_heads * self.head_dim, bias=False)
        self.latent_v_proj = nn.Linear(self.hidden_size, self.num_latent_heads * self.head_dim, bias=False)
        self.latent_o_proj = nn.Linear(self.num_latent_heads * self.head_dim, self.hidden_size, bias=False)
        
        # Rotary embeddings
        self.rotary_emb = LlamaRotaryEmbedding(
            config
            #self.head_dim,
            #max_position_embeddings=self.max_position_embeddings,
            #base=config.rope_theta,
            #scaling_factor=getattr(config, "rope_scaling", None),
        )
        
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        
    def _latent_shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_latent_heads, self.head_dim).transpose(1, 2).contiguous()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        # Standard attention path
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Latent attention path
        latent_query = self.latent_q_proj(hidden_states)  # Project to latent space
        latent_key = self.latent_k_proj(latent_query)  # Project from latent to key
        latent_value = self.latent_v_proj(hidden_states)  # Direct projection to value
        
        # Reshape for multi-head attention
        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)
        
        # Reshape latent tensors
        latent_key = self._latent_shape(latent_key, q_len, bsz)
        latent_value = self._latent_shape(latent_value, q_len, bsz)
        
        # Handle kv caching for inference
        if past_key_value is not None:
            # Separate caches for standard and latent paths
            key_states, value_states, latent_key, latent_value = self._handle_kv_caching(
                key_states, value_states, latent_key, latent_value, past_key_value, q_len
            )
        
        kv_seq_len = key_states.shape[-2]
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        latent_key = apply_rotary_pos_emb(None, latent_key, cos, sin, position_ids)[1]
        
        # Prepare key/value for attention
        key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=0)
        value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=0)
        
        # Standard attention with Flash Attention v3
        attn_output = flash_attn_func(
            query_states.transpose(1, 2),  # [batch_size, seq_len, num_heads, head_dim]
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            causal=True,
        ).transpose(1, 2)
        
        # Latent attention (also with Flash Attention v3)
        latent_output = flash_attn_func(
            query_states.transpose(1, 2),  # Use same queries as standard path
            latent_key.transpose(1, 2),
            latent_value.transpose(1, 2),
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            causal=True,
        ).transpose(1, 2)
        
        # Combine and project outputs
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, self.hidden_size)
        latent_output = latent_output.transpose(1, 2).reshape(bsz, q_len, self.num_latent_heads * self.head_dim)
        
        standard_output = self.o_proj(attn_output)
        latent_output = self.latent_o_proj(latent_output)
        
        # Combine standard and latent outputs
        output = standard_output + latent_output
        
        # Prepare return values
        if use_cache:
            past_key_value = (key_states, value_states, latent_key, latent_value)
        else:
            past_key_value = None
            
        if output_attentions:
            # This implementation doesn't return attention weights currently
            warnings.warn("output_attentions=True is not fully supported for MLA")
            attn_weights = None
        else:
            attn_weights = None
            
        return output, attn_weights, past_key_value
    
    def _handle_kv_caching(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        latent_key: torch.Tensor,
        latent_value: torch.Tensor,
        past_key_value: Tuple[torch.Tensor],
        q_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # past_key_value contains: (key_states, value_states, latent_key, latent_value)
        past_key, past_value, past_latent_key, past_latent_value = past_key_value
        key_states = torch.cat([past_key, key_states], dim=2)
        value_states = torch.cat([past_value, value_states], dim=2)
        latent_key = torch.cat([past_latent_key, latent_key], dim=2)
        latent_value = torch.cat([past_latent_value, latent_value], dim=2)
        return key_states, value_states, latent_key, latent_value

# ============================================================================
# DeepSeek R1 Mixture of Experts (MoE) Implementation
# ============================================================================

class LlamaMoEMLP(nn.Module):
    """
    Mixture of Experts implementation for LLaMA, based on DeepSeek R1 architecture.
    Uses top-k routing to select experts for each token.
    """
    def __init__(self, config: LlamaDeepSeekConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # MoE specific parameters
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        
        # Router for selecting experts
        self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
        
        # Create the experts (each is a standard LLaMA MLP)
        self.experts = nn.ModuleList([
            LlamaMLP(config) for _ in range(self.num_experts)
        ])
        
        # Auxiliary loss for load balancing
        self.router_aux_loss_coef = config.router_aux_loss_coef
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with top-k routing.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size]
            
        Returns:
            Tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Reshape for routing: [batch_size * seq_len, hidden_size]
        x_flat = x.view(-1, hidden_size)
        
        # Get router logits: [batch_size * seq_len, num_experts]
        router_logits = self.router(x_flat)
        
        # Find top-k experts
        routing_weights, selected_experts = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )
        
        # Apply softmax to weights of selected experts only
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Compute load balancing loss
        self._router_aux_loss = self._compute_router_aux_loss(router_logits)
        
        # Prepare output
        final_output = torch.zeros_like(x_flat)
        
        # Dispatch to experts and combine results
        for expert_idx in range(self.num_experts):
            # Find tokens that use this expert
            expert_mask = (selected_experts == expert_idx)
            
            if not expert_mask.any():
                # Skip experts that aren't used
                continue
                
            # Find the batch indices and expert indices where this expert was selected
            batch_indices = torch.nonzero(expert_mask, as_tuple=True)[0]
            expert_weights_indices = torch.nonzero(expert_mask, as_tuple=True)[1]
            
            # Get the corresponding weights for these selections
            expert_weights = routing_weights[batch_indices, expert_weights_indices]
            
            # Select inputs for this expert
            expert_inputs = x_flat[batch_indices]
            
            # Forward through expert
            expert_output = self.experts[expert_idx](expert_inputs)
            
            # Scale by the routing weights
            expert_output = expert_output * expert_weights.unsqueeze(-1)
            
            # Add to final output
            final_output.index_add_(0, batch_indices, expert_output)
            
        # Reshape back to original shape
        final_output = final_output.view(batch_size, seq_len, hidden_size)
        
        return final_output
        
    def _compute_router_aux_loss(self, router_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss to encourage uniform expert utilization.
        
        Args:
            router_logits: Router logits of shape [batch_size * seq_len, num_experts]
            
        Returns:
            Auxiliary loss value
        """
        # Get expert assignment probabilities
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Compute the fraction of tokens routed to each expert
        expert_usage = router_probs.mean(dim=0)
        
        # Target is uniform distribution
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        
        # Use mean squared error between actual and target usage
        aux_loss = F.mse_loss(expert_usage, target_usage) * self.router_aux_loss_coef
        
        return aux_loss

# ============================================================================
# DeepSeek R1 Decoder Layer with MLA and MoE
# ============================================================================

class LlamaDeepSeekDecoderLayer(nn.Module):
    """
    Decoder layer with optional MLA or MoE components.
    Selectively uses either standard attention or MLA, and either standard MLP or MoE.
    """
    def __init__(self, config: LlamaDeepSeekConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        
        # Pre-normalization layers
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Determine if this layer should use MLA
        use_mla_in_this_layer = config.use_mla and (layer_idx in config.mla_layers)
        
        # Determine if this layer should use MoE
        use_moe_in_this_layer = config.use_moe and (layer_idx in config.moe_layers)
        
        # Choose appropriate attention implementation
        if use_mla_in_this_layer:
            self.attention = LlamaMultiheadLatentAttention(config, layer_idx)
            logger.info(f"Layer {layer_idx}: Using Multi-head Latent Attention (MLA)")
        else:
            self.attention = LlamaAttention(config, layer_idx)
            logger.info(f"Layer {layer_idx}: Using standard LLaMA attention")
        
        # Choose appropriate feed-forward implementation
        if use_moe_in_this_layer:
            self.mlp = LlamaMoEMLP(config, layer_idx)
            logger.info(f"Layer {layer_idx}: Using Mixture of Experts (MoE)")
        else:
            self.mlp = LlamaMLP(config)
            logger.info(f"Layer {layer_idx}: Using standard LLaMA MLP")
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Self Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # MLP block (standard or MoE)
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (self_attn_weights,)
            
        if use_cache:
            outputs += (present_key_value,)
            
        return outputs

# ============================================================================
# Full LLaMA-DeepSeek Model Implementation
# ============================================================================

class LlamaDeepSeekModel(LlamaModel):
    """
    LLaMA model with DeepSeek R1 extensions (MLA and MoE).
    Inherits from LlamaModel but replaces selected layers with MLA and MoE.
    """
    config_class = LlamaDeepSeekConfig
    
    def __init__(self, config: LlamaDeepSeekConfig):
        # Initialize with parent class
        PreTrainedModel.__init__(self, config)
        
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([
            LlamaDeepSeekDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens
        
    def set_input_embeddings(self, value):
        self.embed_tokens = value

# CausalLM model with MLA and MoE
class LlamaDeepSeekForCausalLM(LlamaForCausalLM):
    """
    LLaMA causal language model with DeepSeek R1 extensions (MLA and MoE).
    """
    config_class = LlamaDeepSeekConfig
    
    def __init__(self, config: LlamaDeepSeekConfig):
        # Initialize with parent class but replace model type
        PreTrainedModel.__init__(self, config)
        
        # Create the model with our extended architecture
        self.model = LlamaDeepSeekModel(config)
        
        # Create LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights and apply final processing
        self.post_init()

# ============================================================================
# Helper Functions for Weight Transfer
# ============================================================================

def _transfer_attention_weights(src_attn: LlamaAttention, tgt_attn: LlamaAttention):
    """
    Transfer weights from source LlamaAttention to target LlamaAttention.
    Direct copy for standard attention layers.
    
    Args:
        src_attn: Source LlamaAttention module
        tgt_attn: Target LlamaAttention module
    """
    # Copy query, key, value projections
    tgt_attn.q_proj.weight.data.copy_(src_attn.q_proj.weight.data)
    tgt_attn.k_proj.weight.data.copy_(src_attn.k_proj.weight.data)
    tgt_attn.v_proj.weight.data.copy_(src_attn.v_proj.weight.data)
    tgt_attn.o_proj.weight.data.copy_(src_attn.o_proj.weight.data)

def _transfer_attention_to_mla(src_attn: LlamaAttention, tgt_mla: LlamaMultiheadLatentAttention):
    """
    Transfer weights from LlamaAttention to LlamaMultiheadLatentAttention (MLA).
    For MLA, we need special handling for the latent projections.
    
    Args:
        src_attn: Source LlamaAttention module
        tgt_mla: Target LlamaMultiheadLatentAttention module
    """
    # Copy standard attention weights (same as in LlamaAttention)
    tgt_mla.q_proj.weight.data.copy_(src_attn.q_proj.weight.data)
    tgt_mla.k_proj.weight.data.copy_(src_attn.k_proj.weight.data)
    tgt_mla.v_proj.weight.data.copy_(src_attn.v_proj.weight.data)
    tgt_mla.o_proj.weight.data.copy_(src_attn.o_proj.weight.data)
    
    # Initialize latent projections carefully with dimension checking
    logger.info(f"Initializing MLA latent projections with dimensions:")
    logger.info(f"  latent_q_proj: {tgt_mla.latent_q_proj.weight.data.shape}")
    logger.info(f"  latent_k_proj: {tgt_mla.latent_k_proj.weight.data.shape}")
    logger.info(f"  latent_v_proj: {tgt_mla.latent_v_proj.weight.data.shape}")
    logger.info(f"  latent_o_proj: {tgt_mla.latent_o_proj.weight.data.shape}")
    
    # For latent_q_proj: [latent_size, hidden_size]
    # Initialize with small random values
    latent_q_init = torch.randn_like(tgt_mla.latent_q_proj.weight.data) * 0.02
    tgt_mla.latent_q_proj.weight.data.copy_(latent_q_init)
    
    # For latent_k_proj: [num_latent_heads * head_dim, latent_size]
    latent_k_init = torch.randn_like(tgt_mla.latent_k_proj.weight.data) * 0.02
    tgt_mla.latent_k_proj.weight.data.copy_(latent_k_init)
    
    # For latent_v_proj: [num_latent_heads * head_dim, hidden_size]
    # We'll use part of the original v_proj if possible
    # Check dimensions first
    latent_v_shape = tgt_mla.latent_v_proj.weight.data.shape
    src_v_shape = src_attn.v_proj.weight.data.shape
    
    if latent_v_shape[0] <= src_v_shape[0] and latent_v_shape[1] == src_v_shape[1]:
        # Can use a direct slice
        v_proj_slice = src_attn.v_proj.weight.data[:latent_v_shape[0], :]
        tgt_mla.latent_v_proj.weight.data.copy_(v_proj_slice)
    else:
        # Dimensions incompatible, use random init
        logger.warning(f"Cannot directly transfer v_proj weights to latent_v_proj due to shape mismatch: {src_v_shape} vs {latent_v_shape}")
        latent_v_init = torch.randn_like(tgt_mla.latent_v_proj.weight.data) * 0.02
        tgt_mla.latent_v_proj.weight.data.copy_(latent_v_init)
    
    # For latent_o_proj: [hidden_size, num_latent_heads * head_dim]
    # We'll try to adapt from original o_proj
    latent_o_shape = tgt_mla.latent_o_proj.weight.data.shape
    src_o_shape = src_attn.o_proj.weight.data.shape
    
    if latent_o_shape[0] == src_o_shape[0] and latent_o_shape[1] <= src_o_shape[1]:
        # Can use a direct slice
        o_proj_slice = src_attn.o_proj.weight.data[:, :latent_o_shape[1]]
        tgt_mla.latent_o_proj.weight.data.copy_(o_proj_slice)
    else:
        # Dimensions incompatible, use random init
        logger.warning(f"Cannot directly transfer o_proj weights to latent_o_proj due to shape mismatch: {src_o_shape} vs {latent_o_shape}")
        latent_o_init = torch.randn_like(tgt_mla.latent_o_proj.weight.data) * 0.02
        tgt_mla.latent_o_proj.weight.data.copy_(latent_o_init)

def _transfer_mlp_weights(src_mlp: LlamaMLP, tgt_mlp: LlamaMLP):
    """
    Transfer weights from source LlamaMLP to target LlamaMLP.
    Direct copy for standard MLP layers.
    
    Args:
        src_mlp: Source LlamaMLP module
        tgt_mlp: Target LlamaMLP module
    """
    tgt_mlp.gate_proj.weight.data.copy_(src_mlp.gate_proj.weight.data)
    tgt_mlp.down_proj.weight.data.copy_(src_mlp.down_proj.weight.data)
    tgt_mlp.up_proj.weight.data.copy_(src_mlp.up_proj.weight.data)

def _transfer_mlp_to_moe(src_mlp: LlamaMLP, tgt_moe: LlamaMoEMLP):
    """
    Transfer weights from LlamaMLP to LlamaMoEMLP (Mixture of Experts).
    For MoE, we need to initialize the experts and the router.
    
    Args:
        src_mlp: Source LlamaMLP module
        tgt_moe: Target LlamaMoEMLP module
    """
    # Log expert dimensions for debugging
    expert0 = tgt_moe.experts[0]
    logger.info(f"MoE dimensions:")
    logger.info(f"  Source gate_proj: {src_mlp.gate_proj.weight.data.shape}")
    logger.info(f"  Source up_proj: {src_mlp.up_proj.weight.data.shape}")
    logger.info(f"  Source down_proj: {src_mlp.down_proj.weight.data.shape}")
    logger.info(f"  Expert gate_proj: {expert0.gate_proj.weight.data.shape}")
    logger.info(f"  Expert up_proj: {expert0.up_proj.weight.data.shape}")
    logger.info(f"  Expert down_proj: {expert0.down_proj.weight.data.shape}")
    
    # Initialize the router with small random weights
    router_init = torch.randn_like(tgt_moe.router.weight.data) * 0.01
    tgt_moe.router.weight.data.copy_(router_init)
    
    # For each expert, safely transfer or initialize weights
    for i in range(tgt_moe.num_experts):
        expert = tgt_moe.experts[i]
        
        # For the first expert, try to copy from source MLP
        if i == 0:
            # Copy weights safely with dimension checking
            _safe_copy_mlp_weights(src_mlp, expert)
        else:
            # For other experts, add noise to expert 0
            # but first check if weights were initialized
            if hasattr(expert0.gate_proj, 'weight') and expert0.gate_proj.weight is not None:
                expert.gate_proj.weight.data.copy_(expert0.gate_proj.weight.data + torch.randn_like(expert0.gate_proj.weight.data) * 0.01)
                expert.up_proj.weight.data.copy_(expert0.up_proj.weight.data + torch.randn_like(expert0.up_proj.weight.data) * 0.01)
                expert.down_proj.weight.data.copy_(expert0.down_proj.weight.data + torch.randn_like(expert0.down_proj.weight.data) * 0.01)
            else:
                # Expert 0 wasn't properly initialized, initialize this expert with random weights
                logger.warning(f"Expert 0 not properly initialized. Initializing expert {i} with random weights.")
                expert.gate_proj.weight.data.normal_(mean=0.0, std=0.02)
                expert.up_proj.weight.data.normal_(mean=0.0, std=0.02)
                expert.down_proj.weight.data.normal_(mean=0.0, std=0.02)

def _safe_copy_mlp_weights(src_mlp: LlamaMLP, tgt_mlp: LlamaMLP):
    """
    Safely copy MLP weights with dimension checking.
    
    Args:
        src_mlp: Source LlamaMLP module
        tgt_mlp: Target LlamaMLP module
    """
    # Check and copy gate_proj weights
    if src_mlp.gate_proj.weight.data.shape == tgt_mlp.gate_proj.weight.data.shape:
        tgt_mlp.gate_proj.weight.data.copy_(src_mlp.gate_proj.weight.data)
    else:
        logger.warning(f"Cannot copy gate_proj weights due to shape mismatch: {src_mlp.gate_proj.weight.data.shape} vs {tgt_mlp.gate_proj.weight.data.shape}")
        tgt_mlp.gate_proj.weight.data.normal_(mean=0.0, std=0.02)
    
    # Check and copy up_proj weights
    if src_mlp.up_proj.weight.data.shape == tgt_mlp.up_proj.weight.data.shape:
        tgt_mlp.up_proj.weight.data.copy_(src_mlp.up_proj.weight.data)
    else:
        logger.warning(f"Cannot copy up_proj weights due to shape mismatch: {src_mlp.up_proj.weight.data.shape} vs {tgt_mlp.up_proj.weight.data.shape}")
        tgt_mlp.up_proj.weight.data.normal_(mean=0.0, std=0.02)
    
    # Check and copy down_proj weights
    if src_mlp.down_proj.weight.data.shape == tgt_mlp.down_proj.weight.data.shape:
        tgt_mlp.down_proj.weight.data.copy_(src_mlp.down_proj.weight.data)
    else:
        logger.warning(f"Cannot copy down_proj weights due to shape mismatch: {src_mlp.down_proj.weight.data.shape} vs {tgt_mlp.down_proj.weight.data.shape}")
        tgt_mlp.down_proj.weight.data.normal_(mean=0.0, std=0.02)
        
# ============================================================================
# FSDP Utilities 
# ============================================================================

def prepare_model_for_fsdp(model, mixed_precision=True):
    """
    Prepare the model for FSDP training by wrapping it appropriately.
    
    Args:
        model: The model to prepare
        mixed_precision: Whether to use mixed precision
        
    Returns:
        The wrapped model ready for FSDP training
    """
    # Ensure model is on CPU before FSDP wrapping
    model.cpu()
    
    # Define wrapping policy for transformer layers
    transformer_layer_cls = (
        LlamaDeepSeekDecoderLayer,
        LlamaMultiheadLatentAttention,
        LlamaMoEMLP,
    )
    
    # Wrap policy to identify transformer layers
    wrap_policy = transformer_auto_wrap_policy(transformer_layer_cls)
    
    # Set mixed precision configuration
    if mixed_precision:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    else:
        mixed_precision_policy = None
    
    # Initialize FSDP
    model = FSDP(
        model,
        auto_wrap_policy=wrap_policy,
        mixed_precision=mixed_precision_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    )
    
    return model

def freeze_original_layers(model, mla_layers, moe_layers):
    """
    Freeze the original LLaMA layers that were not converted to MLA or MoE.
    This is useful for efficient fine-tuning where we only train the new components.
    
    Args:
        model: The converted model
        mla_layers: List of layer indices converted to MLA
        moe_layers: List of layer indices converted to MoE
    """
    modified_layers = set(mla_layers + moe_layers)
    
    # Freeze embeddings
    for param in model.model.embed_tokens.parameters():
        param.requires_grad = False
    
    # Freeze final norm
    for param in model.model.norm.parameters():
        param.requires_grad = False
    
    # Selectively freeze decoder layers
    for i, layer in enumerate(model.model.layers):
        if i not in modified_layers:
            # Freeze entire unchanged layer
            for param in layer.parameters():
                param.requires_grad = False
        else:
            # For modified layers, freeze only unchanged components
            for param in layer.input_layernorm.parameters():
                param.requires_grad = False
                
            for param in layer.post_attention_layernorm.parameters():
                param.requires_grad = False
            
            if i in mla_layers:
                # For MLA layers, keep original attention frozen
                pass  # We'll train all MLA parameters
            else:
                # Freeze attention in non-MLA layers
                for param in layer.attention.parameters():
                    param.requires_grad = False
                    
            if i in moe_layers:
                # For MoE layers, keep MoE parameters trainable
                pass  # We'll train all MoE parameters
            else:
                # Freeze MLP in non-MoE layers
                for param in layer.mlp.parameters():
                    param.requires_grad = False
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Model has {total_params:,} total parameters")
    logger.info(f"Model has {trainable_params:,} trainable parameters ({trainable_params/total_params:.2%})")
    
    return model

# ============================================================================
# Model Loading Utilities
# ============================================================================

def load_converted_model(model_path, device_map="auto"):
    """
    Load a previously converted LlamaDeepSeek model.
    
    Args:
        model_path: Path to the saved model
        device_map: Device mapping strategy (auto, balanced, etc.)
        
    Returns:
        The loaded model and tokenizer
    """
    # Register the custom model architecture with the transformers library
    if not hasattr(transformers.models, "llama_deepseek"):
        # Create a module for the custom model
        transformers.models.llama_deepseek = type('llama_deepseek', (), {})()
        
        # Register the model classes
        transformers.models.llama_deepseek.configuration_llama_deepseek = LlamaDeepSeekConfig
        transformers.models.llama_deepseek.modeling_llama_deepseek = LlamaDeepSeekForCausalLM
        
        # Register the model type
        transformers.models.auto.configuration_auto.CONFIG_MAPPING.update(
            {"llama_deepseek": LlamaDeepSeekConfig}
        )
        transformers.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING.update(
            {LlamaDeepSeekConfig: LlamaDeepSeekForCausalLM}
        )
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    
    return model, tokenizer

# ============================================================================
# Conversion Utilities
# ============================================================================

def convert_llama_to_deepseek(
    llama_model_path: str, 
    output_dir: str,
    mla_layers: Optional[List[int]] = None,
    moe_layers: Optional[List[int]] = None,
    num_experts: int = 8,
    num_experts_per_tok: int = 2,
):
    """
    Convert LLaMA-3.1 8B model to a DeepSeek R1 style model with MLA and MoE.
    
    Args:
        llama_model_path: Path to the original LLaMA-3.1 8B model
        output_dir: Directory to save the converted model
        mla_layers: List of layer indices to convert to MLA (None uses default)
        moe_layers: List of layer indices to convert to MoE (None uses default)
        num_experts: Number of experts in MoE layers
        num_experts_per_tok: Number of active experts per token
    """
    logger.info(f"Loading original LLaMA model from {llama_model_path}")
    
    # Load the original model and configuration
    orig_model = AutoModelForCausalLM.from_pretrained(
        llama_model_path, 
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    orig_config = orig_model.config

    print (orig_config)
    
    # Extract key parameters from original config
    vocab_size = orig_config.vocab_size
    hidden_size = orig_config.hidden_size
    intermediate_size = orig_config.intermediate_size
    num_hidden_layers = orig_config.num_hidden_layers
    num_attention_heads = orig_config.num_attention_heads
    num_key_value_heads = orig_config.num_key_value_heads
    max_position_embeddings = orig_config.max_position_embeddings
    rms_norm_eps = orig_config.rms_norm_eps
    rope_theta = orig_config.rope_theta
    rope_scaling = orig_config.rope_scaling
    
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    # Set default MLA and MoE layers if not provided
    # We select layers strategically across the network
    if mla_layers is None:
        # Use middle-alternate-layers 28% for MLA (9/32 for 8B)
        # Distribute across the network with focus on middle layers
        mla_layers = [8, 10, 12, 14, 16, 18, 20, 22, 24]

    '''
    Alternate Layers Approach: Converting every other layer (layers 8, 10, 12, 14, 16, 18, 20, 22, 24) offers several advantages:
        - Gradient stability: Alternating patterns help maintain more stable gradient flow
        - Architectural diversity: Provides a mix of processing mechanisms that can be complementary
        - Lower implementation risk: More conservative approach that preserves some of the original architecture
        - Easier performance isolation: Simplifies attribution of performance changes to specific modifications
    '''
        
    if moe_layers is None:
        # Use approximately 12.5% of layers toward final layers for MoE (4/32 for 8B)
        # Distribute across the network with focus towards final layers
        moe_layers = [11, 15, 19, 23]

    '''
    Rationale:
    
    Early enough to influence mid-network processing
    Located at a critical middle point in the reasoning chain
    Can specialize in different reasoning approaches after significant attention refinement
    Late enough to influence final output generation
    Provides expert routing for response formatting after all MLA processing is complete
    '''
    # -----------------------------------------------------------------------------------------------------------------------------------------------------------
    
    # Check for overlapping MLA and MoE layers
    if set(mla_layers).intersection(set(moe_layers)):
        raise ValueError("MLA and MoE layers cannot overlap. Please provide non-overlapping layer indices.")
    
    # Verify layer indices are valid
    max_layer_idx = num_hidden_layers - 1
    if max(mla_layers + moe_layers) > max_layer_idx:
        raise ValueError(f"Layer indices must be less than {num_hidden_layers}. Found layer index: {max(mla_layers + moe_layers)}")
    
    logger.info(f"MLA will be applied to layers: {mla_layers}")
    logger.info(f"MoE will be applied to layers: {moe_layers}")
    
    # Create new DeepSeek configuration
    new_config = LlamaDeepSeekConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        use_mla=True,
        mla_layers=mla_layers,
        num_latent_heads=16,  # DeepSeek R1 default
        latent_size=1024,     # DeepSeek R1 default
        use_moe=True,
        moe_layers=moe_layers,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    
    # Create new model with our extended architecture
    logger.info("Creating new DeepSeek model with MLA and MoE")
    new_model = LlamaDeepSeekForCausalLM(new_config)
    
    # Begin weight transfer
    logger.info("Starting weight transfer from LLaMA to DeepSeek model")
    
    # Transfer embedding weights
    logger.info("Transferring embedding weights")
    new_model.model.embed_tokens.weight.data.copy_(orig_model.model.embed_tokens.weight.data)
    new_model.lm_head.weight.data.copy_(orig_model.lm_head.weight.data)
    
    # Transfer layer norm weights
    logger.info("Transferring final layer norm weights")
    new_model.model.norm.weight.data.copy_(orig_model.model.norm.weight.data)
    
    # Transfer decoder layer weights with progress bar
    logger.info("Transferring decoder layer weights")
    for i in tqdm(range(num_hidden_layers), desc="Transferring layers"):
        # Determine if this layer uses MLA or MoE
        use_mla = i in mla_layers
        use_moe = i in moe_layers
        
        # Get source and target layers
        src_layer = orig_model.model.layers[i]
        tgt_layer = new_model.model.layers[i]
        
        # Transfer layer norms (same for all layer types)
        tgt_layer.input_layernorm.weight.data.copy_(src_layer.input_layernorm.weight.data)
        tgt_layer.post_attention_layernorm.weight.data.copy_(src_layer.post_attention_layernorm.weight.data)
        
        # Attention weights transfer
        if use_mla:
            # Transfer weights to MLA
            _transfer_attention_to_mla(src_layer.self_attn, tgt_layer.attention)
        else:
            # Standard attention transfer (direct copy)
            _transfer_attention_weights(src_layer.self_attn, tgt_layer.attention)
        
        # MLP weights transfer
        if use_moe:
            # Transfer weights to MoE
            _transfer_mlp_to_moe(src_layer.mlp, tgt_layer.mlp)
        else:
            # Standard MLP transfer (direct copy)
            _transfer_mlp_weights(src_layer.mlp, tgt_layer.mlp)
    
    # Free up memory
    del orig_model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save the converted model
    logger.info(f"Saving converted model to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save configuration
    new_config.save_pretrained(output_dir)
    
    # Save model weights
    new_model.save_pretrained(
        output_dir,
        save_function=torch.save,
        safe_serialization=False,  # Use standard PyTorch serialization
        max_shard_size="10GB",     # Shard large models
    )
    
    # Copy tokenizer files
    logger.info("Copying tokenizer files")
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
    tokenizer.save_pretrained(output_dir)
    
    logger.info("Conversion completed successfully!")
    logger.info(f"Model successfully saved to {output_dir}")
    logger.info("You can load this model with: from transformers import AutoModelForCausalLM, AutoTokenizer")
    logger.info(f'model = AutoModelForCausalLM.from_pretrained("{output_dir}")')
    
    return new_model, new_config