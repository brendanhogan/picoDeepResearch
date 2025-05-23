"""
Core module for loading and configuring language models.

Supports:
- HuggingFace models (local or remote)
- OpenAI API models (GPT series)
- Anthropic API models (Claude series)

Each model is wrapped in a ModelInterface that provides consistent generation methods.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizerBase
from model_interface import ModelInterface, HuggingFaceModel, OpenAIModel, AnthropicModel
from typing import Union, Dict, Any
from importlib.util import find_spec


def is_liger_available() -> bool:
    """Check if Liger kernel optimization is available."""
    return find_spec("liger_kernel") is not None

def get_model(model_name: str, model_kwargs: Union[Dict[str, Any], None] = None) -> Any:
    """
    Load a language model with optimized settings.
    
    Args:
        model_name: Name or path of the model to load
        model_kwargs: Optional model configuration overrides
        
    Returns:
        Loaded model with optimized settings for inference
    """
    if model_kwargs is None:
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
            device_map="auto",
        )
    if False:  # TODO: Liger optimization not working
        print("Using Liger kernel")
        from liger_kernel.transformers import AutoLigerKernelForCausalLM
        return AutoLigerKernelForCausalLM.from_pretrained(model_name, **model_kwargs)
    else:
        print("Using AutoModelForCausalLM")
        return AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

def get_llm_tokenizer(model_name: str, device: str) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Load and configure a language model with its tokenizer.
    
    Args:
        model_name: Name or path of the model to load
        device: Target device ('cpu' or 'cuda')
        
    Returns:
        Tuple of (model, tokenizer) with consistent padding configuration
    """
    model = get_model(model_name)
    
    # Configure tokenizer and model for consistent padding
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    
    return model, tokenizer

def get_judge_model(model_name: str, device: str) -> ModelInterface:
    """
    Create a judge model for evaluating report quality.
    
    Args:
        model_name: Model identifier (HF model name or API model name)
        device: Target device ('cpu' or 'cuda')
        
    Returns:
        ModelInterface configured for judging report quality
    """
    if model_name.startswith(('gpt-', 'claude-')):
        if model_name.startswith('gpt-'):
            return OpenAIModel(model_name)
        else:
            return AnthropicModel(model_name)
    else:
        model, tokenizer = get_llm_tokenizer(model_name, device)
        return HuggingFaceModel(model, tokenizer, device)

def get_compare_model(model_name: str, device: str) -> ModelInterface:
    """
    Create a model for generating comparison reports.
    
    Args:
        model_name: Model identifier (HF model name or API model name)
        device: Target device ('cpu' or 'cuda')
        
    Returns:
        ModelInterface configured for generating comparison reports
    """
    if model_name.startswith(('gpt-', 'claude-')):
        if model_name.startswith('gpt-'):
            return OpenAIModel(model_name)
        else:
            return AnthropicModel(model_name)
    else:
        model, tokenizer = get_llm_tokenizer(model_name, device)
        return HuggingFaceModel(model, tokenizer, device)
