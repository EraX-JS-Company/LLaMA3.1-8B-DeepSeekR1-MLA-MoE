#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to load and test the custom LLaMA-DeepSeek model.
This script imports model classes directly from the conversion module
to ensure consistency between conversion and loading.
"""

import os
import sys
import logging
import argparse
import torch
import json
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Tuple, List, Dict, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Import model classes directly from the conversion script
# This ensures consistency between conversion and loading
from llama_deepseek_convert import (
    LlamaDeepSeekConfig,
    LlamaMultiheadLatentAttention,
    LlamaMoEMLP,
    LlamaDeepSeekDecoderLayer,
    LlamaDeepSeekModel,
    LlamaDeepSeekForCausalLM,
)

def load_config_directly(config_path):
    """
    Load configuration directly from JSON file to inspect it.
    
    Args:
        config_path: Path to the config.json file
        
    Returns:
        Dictionary containing configuration
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def update_model_config(model_path):
    """
    Update the model's config.json to use our custom model type.
    
    Args:
        model_path: Path to the model directory
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}")
        return
    
    # Load the existing config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Update model_type and architectures
    config["model_type"] = "llama_deepseek"
    config["architectures"] = ["LlamaDeepSeekForCausalLM"]
    
    # Ensure MLA and MoE layers are defined properly
    if "mla_layers" not in config:
        config["mla_layers"] = [8, 10, 12, 14, 16, 18, 20, 22, 24]
    if "moe_layers" not in config:
        config["moe_layers"] = [11, 15, 19, 23]
    
    # Add other required fields
    config["use_mla"] = True
    config["use_moe"] = True
    config["num_experts"] = 8
    config["num_experts_per_tok"] = 2
    config["num_latent_heads"] = 16
    config["latent_size"] = 1024
    
    # Save the updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Updated config at {config_path} to use custom model type")

def register_llama_deepseek_model():
    """
    Register the custom LlamaDeepSeek model with the transformers library.
    This is necessary to load the model using AutoModelForCausalLM.
    """
    # Check if model is already registered
    if "llama_deepseek" in transformers.models.auto.configuration_auto.CONFIG_MAPPING:
        logger.info("LlamaDeepSeekConfig already registered in CONFIG_MAPPING")
        return
    
    # Register the config with the auto mapping
    transformers.models.auto.configuration_auto.CONFIG_MAPPING.register("llama_deepseek", LlamaDeepSeekConfig)
    logger.info("Registered LlamaDeepSeekConfig with CONFIG_MAPPING")
    
    # Register for causal LM
    transformers.models.auto.modeling_auto.MODEL_FOR_CAUSAL_LM_MAPPING.register(LlamaDeepSeekConfig, LlamaDeepSeekForCausalLM)
    logger.info("Registered LlamaDeepSeekForCausalLM with MODEL_FOR_CAUSAL_LM_MAPPING")
    
    # Set up model classes as a module
    if not hasattr(transformers.models, "llama_deepseek"):
        # Create module structure
        transformers.models.llama_deepseek = type('module', (), {
            '__file__': __file__,
            '__package__': 'transformers.models.llama_deepseek',
            '__path__': [],
            '__spec__': None,
            'configuration_llama_deepseek': type('module', (), {
                'LlamaDeepSeekConfig': LlamaDeepSeekConfig
            }),
            'modeling_llama_deepseek': type('module', (), {
                'LlamaMultiheadLatentAttention': LlamaMultiheadLatentAttention,
                'LlamaMoEMLP': LlamaMoEMLP,
                'LlamaDeepSeekDecoderLayer': LlamaDeepSeekDecoderLayer,
                'LlamaDeepSeekModel': LlamaDeepSeekModel, 
                'LlamaDeepSeekForCausalLM': LlamaDeepSeekForCausalLM
            })
        })
        
        logger.info("Created llama_deepseek module with model classes")

def load_model(model_path, device_map="auto"):
    """
    Load the LLaMA-DeepSeek model properly by registering it first.
    
    Args:
        model_path: Path to the model directory
        device_map: Device mapping strategy
        
    Returns:
        The loaded model and tokenizer
    """
    # Update the model config to ensure it has the right type
    update_model_config(model_path)
    
    # Register our custom model type
    register_llama_deepseek_model()
    
    # First, let's inspect the config.json directly to see what's in it
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            raw_config = load_config_directly(config_path)
            logger.info(f"Raw config model_type: {raw_config.get('model_type')}")
            logger.info(f"MLA layers: {raw_config.get('mla_layers')}")
            logger.info(f"MoE layers: {raw_config.get('moe_layers')}")
            
            if 'architectures' in raw_config:
                logger.info(f"Config contains architectures: {raw_config['architectures']}")
        except Exception as e:
            logger.warning(f"Error inspecting config directly: {e}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info(f"Loaded tokenizer from {model_path}")
    
    try:
        # Load the model with our registered class
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map=device_map,
            trust_remote_code=True
        )
        logger.info(f"Successfully loaded model from {model_path}")
        
    except Exception as primary_error:
        logger.warning(f"Primary loading method failed: {primary_error}")
        
        try:
            # Try alternative loading approach
            logger.info("Trying alternative approach using load_converted_model")
            
            # Import the function directly from conversion script
            from llama_deepseek_convert import load_converted_model
            model, tokenizer = load_converted_model(model_path, device_map)
            logger.info(f"Successfully loaded model using load_converted_model")
            
        except Exception as fallback_error:
            logger.error(f"Alternative loading also failed: {fallback_error}")
            raise RuntimeError(f"Could not load model: {primary_error}. Alternative also failed: {fallback_error}")
    
    return model, tokenizer

def test_model_generation(model, tokenizer, prompt, max_new_tokens=50):
    """
    Test text generation with the model.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Text prompt for generation
        max_new_tokens: Maximum number of tokens to generate
    """
    # Format the prompt
    logger.info(f"Testing model with prompt: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"Generated text: {generated_text}")
    
    return generated_text

def inspect_model_structure(model):
    """
    Inspect the model structure to see if MLA and MoE components are present.
    Uses knowledge of the model classes from the conversion script.
    
    Args:
        model: The loaded model
        
    Returns:
        Dictionary with inspection results
    """
    results = {
        "model_type": getattr(model.config, "model_type", "unknown"),
        "config_class": model.config.__class__.__name__,
        "model_class": model.__class__.__name__,
        "has_mla_fields": False,
        "has_moe_fields": False,
        "mla_layers": [],
        "moe_layers": [],
        "mla_layers_found": [],
        "moe_layers_found": []
    }
    
    # Check config for MLA and MoE fields
    if hasattr(model.config, "use_mla"):
        results["has_mla_fields"] = True
        results["mla_layers"] = getattr(model.config, "mla_layers", [])
    
    if hasattr(model.config, "use_moe"):
        results["has_moe_fields"] = True
        results["moe_layers"] = getattr(model.config, "moe_layers", [])
    
    # Display model structure
    results["model_repr"] = repr(model)
    
    # Try to inspect layers if possible
    try:
        layers = model.model.layers
        
        for i, layer in enumerate(layers):
            layer_info = {"index": i, "type": layer.__class__.__name__}
            
            # Try to detect MLA
            if isinstance(layer.attention, LlamaMultiheadLatentAttention):
                results["mla_layers_found"].append(i)
                layer_info["has_mla"] = True
            
            # Try to detect MoE
            if isinstance(layer.mlp, LlamaMoEMLP):
                results["moe_layers_found"].append(i)
                layer_info["has_moe"] = True
            
            results[f"layer_{i}"] = layer_info
    
    except Exception as e:
        logger.warning(f"Could not inspect model layers: {e}")
        results["layer_inspection_error"] = str(e)
    
    return results

def main():
    """
    Main function to load and test the model.
    """
    parser = argparse.ArgumentParser(description="Load and test LLaMA-DeepSeek model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to the converted model directory"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Write a short poem about artificial intelligence.",
        help="Text prompt for testing generation"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=100,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device mapping strategy"
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Only inspect model structure without generation"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default="erax-ai/LLaMA-3.18B-DeepSeekR1-MLA-MoE-8B",
        help="Model ID for Hugging Face Hub"
    )
    
    args = parser.parse_args()
    
    try:
        # Load the model and tokenizer
        model, tokenizer = load_model(args.model_path, args.device_map)
        
        # Inspect model structure
        logger.info("\nInspecting model structure...")
        inspection_results = inspect_model_structure(model)
        
        logger.info("\nModel Structure Inspection Results:")
        for key, value in inspection_results.items():
            if key.startswith("layer_"):
                continue  # Skip detailed layer info in main output
            logger.info(f"{key}: {value}")
        
        # Show MLA and MoE layers detected
        logger.info("\nMLA layers detected: " + str(inspection_results.get("mla_layers_found", [])))
        logger.info("MoE layers detected: " + str(inspection_results.get("moe_layers_found", [])))

        # Push to Hub if requested
        if args.push_to_hub:
            logger.info(f"\nPushing model to Hugging Face Hub: {args.hub_model_id}")
            model.push_to_hub(args.hub_model_id)
            tokenizer.push_to_hub(args.hub_model_id)
        
        # Test generation if not in inspect-only mode
        if not args.inspect_only:
            test_model_generation(model, tokenizer, args.prompt, args.max_new_tokens)
        
        logger.info("\nTest completed successfully!")
        
    except Exception as e:
        logger.error(f"Error loading or testing model: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
