"""
Core interface for language model interactions.

Provides a unified interface for:
- HuggingFace models (local inference)
- OpenAI models (API-based)
- Anthropic models (API-based)

Each implementation handles model-specific details while exposing consistent generation methods.
"""

import time
import torch
import openai
import anthropic
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from transformers import PreTrainedModel, PreTrainedTokenizerBase

class ModelInterface(ABC):
    """
    Base interface for language model interactions.
    
    All model implementations must provide a generate method that accepts
    system and user prompts and returns generated text.
    """
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """
        Generate text from the model.
        
        Args:
            system_prompt: Model instructions/context
            user_prompt: User's input text
            **kwargs: Model-specific generation parameters
            
        Returns:
            Generated text response
        """
        pass

class HuggingFaceModel(ModelInterface):
    """
    Local model implementation using HuggingFace transformers.
    
    Handles:
    - Model and tokenizer initialization
    - Chat template formatting
    - Device placement
    - Token generation and decoding
    """
    
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # Format messages in chat template
        prompt = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
        prompt_text = self.tokenizer.apply_chat_template(prompt, tokenize=False, enable_thinking=True)
        
        # Prepare input tensors
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
            padding_side="left"
        ).to(self.device)
        
        # Generate response
        outputs = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            **kwargs
        )
        
        # Extract and decode new tokens
        response = self.tokenizer.decode(
            outputs[0, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()

class OpenAIModel(ModelInterface):
    """
    OpenAI API model implementation.
    
    Features:
    - Automatic parameter translation
    - Exponential backoff retry logic
    - Error handling
    """
    
    def __init__(self, model_name: str):
        self.client = openai.OpenAI()
        self.model_name = model_name
        
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        # Map HuggingFace parameters to OpenAI parameters
        openai_kwargs = {}
        if 'max_new_tokens' in kwargs:
            openai_kwargs['max_tokens'] = kwargs.pop('max_new_tokens')
        if 'temperature' in kwargs:
            openai_kwargs['temperature'] = kwargs.pop('temperature')
        if 'top_p' in kwargs:
            openai_kwargs['top_p'] = kwargs.pop('top_p')
            
        # Retry logic with exponential backoff
        max_retries = 5
        base_delay = 1  # Initial delay in seconds
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    **openai_kwargs
                )
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    raise e
                    
                # Exponential backoff: 1, 2, 4, 8, 16 seconds
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue

class AnthropicModel(ModelInterface):
    """
    Anthropic API model implementation.
    
    Features:
    - Simplified parameter handling
    - Combined system/user prompt formatting
    - Direct response extraction
    """
    
    def __init__(self, model_name: str):
        self.client = anthropic.Anthropic()
        self.model_name = model_name
        
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=kwargs.get('max_tokens', 4096),
            messages=[
                {"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"}
            ]
        )
        return response.content[0].text.strip() 