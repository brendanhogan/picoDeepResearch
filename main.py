"""
picoDeepResearch: A lightweight implementation of web-based research training.

Core functionality:
1. Web search and information gathering
2. Report generation and compilation
3. Quality evaluation using rubrics
4. Model improvement via GRPO (Generalized Reinforcement Policy Optimization)

The system trains an agent to perform research tasks by:
- Conducting iterative web searches
- Synthesizing findings into structured reports
- Evaluating report quality against benchmarks
- Optimizing performance through reinforcement learning
"""

import gc
import os
import json
import yaml
import torch
import logging
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig

import llms
import utils
import tools
import arenas
import evaluator
import arenastages 


def generate_completions_with_tools(args, arena, model, tokenizer, prompt, num_rollouts, device):
    """
    Generate model completions with multi-turn tool interaction support.
    
    Handles:
    - Multi-turn tool use with response masking
    - Parallel generation across multiple rollouts
    - Tool call logging and tracking
    
    Args:
        args: Generation configuration
        arena: Arena instance with tools
        model: Language model for generation
        tokenizer: Model tokenizer
        prompt: Input prompt text
        num_rollouts: Number of parallel generations
        device: Computation device
        
    Returns:
        Dictionary containing:
        - full_ids: Tensor of token IDs for all rollouts [num_rollouts, seq_len]
        - full_attention_mask: Attention mask tensor [num_rollouts, seq_len]
        - generated_tokens_mask: Mask indicating which tokens were generated vs tool responses [num_rollouts, seq_len]
        - final_texts: List of decoded text for each rollout
        - tool_call_logs: List of tool call and response logs for each rollout
        - prompt_text: Original formatted prompt text
    """
    # Format prompt with system and user messages
    prompt = [{'role': 'user', 'content': prompt}]
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, enable_thinking=True)


    # We basically just want to track: 
    # full_ids - prompt + all responsed 
    # full_attention_mask - just 1's up until padding 
    # generated_token_mask - exclude the prompt and stuff the tool generated 

    # Initialize tracking for each rollout
    full_ids = [[] for _ in range(num_rollouts)]
    full_attention_mask = [[] for _ in range(num_rollouts)]
    generated_tokens_mask = [[] for _ in range(num_rollouts)]


    completion_ids = [[] for _ in range(num_rollouts)]
    completion_attention_mask = [[] for _ in range(num_rollouts)]
    completion_loss_mask = [[] for _ in range(num_rollouts)]

    tool_call_logs = [[] for _ in range(num_rollouts)]
    current_prompts = [prompt_text] * num_rollouts
    num_tool_calls = [0] * num_rollouts
    active_rollouts = list(range(num_rollouts))

    while active_rollouts:
        # Tokenize current prompts for active rollouts
        prompt_inputs = tokenizer(
            [current_prompts[i] for i in active_rollouts],
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=True
        )

        prompt_ids = prompt_inputs["input_ids"].to(device)
        prompt_mask = prompt_inputs["attention_mask"].to(device)

        # Initialize full_ids and full_attention_mask with prompt IDs and mask if empty
        for i, rollout_idx in enumerate(active_rollouts):
            if not full_ids[rollout_idx]:
                original_prompt_length = prompt_ids.shape[1]
                full_ids[rollout_idx] = prompt_ids[i].tolist()
                full_attention_mask[rollout_idx] = prompt_mask[i].tolist()
                generated_tokens_mask[rollout_idx] = [0] * len(prompt_mask[i].tolist())
        
        
        
        # Configure generation parameters
        generation_config = GenerationConfig(
            do_sample=True, 
            temperature=args.temperature,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id, 
            stop_strings=["</tool>", "</answer>"]
        )

        # Generate completions for active rollouts
        completion_ids = model.generate(
            prompt_ids,
            attention_mask=prompt_mask,
            generation_config=generation_config,
            max_new_tokens=32768, 
            tokenizer=tokenizer)

        # Process each active rollout
        new_active_rollouts = []
        for idx, rollout_idx in enumerate(active_rollouts):
            # Get current prompt length to extract only new tokens
            current_prompt_ids = tokenizer.encode(current_prompts[rollout_idx], return_tensors="pt").to(device)
            current_prompt_len = current_prompt_ids.size(1)
            
            # Extract only the new tokens from this generation step
            new_tokens_potentially_with_padding = completion_ids[idx, current_prompt_len:]

            # Remove padding tokens (EOS tokens) from the end
            eos_mask = (new_tokens_potentially_with_padding == tokenizer.eos_token_id)
            if eos_mask.any():
                eos_idx = eos_mask.nonzero()[0][0]
                new_tokens = new_tokens_potentially_with_padding[:eos_idx]
            else:
                new_tokens = new_tokens_potentially_with_padding
            completion_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

            # Handle tool calls
            if completion_text.strip().endswith("</tool>"):
                # Extract tool call XML
                tool_call_start = completion_text.rfind("<tool")
                tool_call_xml = completion_text[tool_call_start:]
                
                # Check tool call limits
                if num_tool_calls[rollout_idx] >= args.max_tool_interactions:
                    tool_response = "<tool_response error='Maximum number of tool calls reached' />"
                else:
                    tool_response = arena.tools.execute_xml_tool_call(tool_call_xml)
                    num_tool_calls[rollout_idx] += 1
                
                # Log tool interaction
                tool_call_logs[rollout_idx].append(f"Tool call: {tool_call_xml}")
                tool_call_logs[rollout_idx].append(f"Tool response: {tool_response}")
                
                # Create masks for loss computation
                # Model output (including tool call) gets mask=1
                model_output_mask = torch.ones_like(new_tokens)
                
                # Tool response gets mask=0
                tool_response_ids = tokenizer.encode(tool_response, add_special_tokens=False, return_tensors="pt").to(device)
                tool_response_mask = torch.zeros_like(tool_response_ids)
                
                # Update history with model output and tool response
                full_ids[rollout_idx].extend(new_tokens.flatten().tolist()) # Just adding all new_tokens processed 
                full_ids[rollout_idx].extend(tool_response_ids.flatten().tolist()) # AND tool response tokens
                full_attention_mask[rollout_idx].extend(torch.ones_like(new_tokens).flatten().tolist()) 
                full_attention_mask[rollout_idx].extend(torch.ones_like(tool_response_ids).flatten().tolist()) # For attention - we *do* want to attend to tool responses
                
                # But we need to track - of the new tokens being added on - are they generated or from tool responses? 
                generated_tokens_mask[rollout_idx].extend(torch.ones_like(new_tokens).flatten().tolist()) 
                generated_tokens_mask[rollout_idx].extend(torch.zeros_like(tool_response_ids).flatten().tolist()) 
                # Update prompt with completion and response
                current_prompts[rollout_idx] = current_prompts[rollout_idx] + completion_text + tool_response
                new_active_rollouts.append(rollout_idx)

            else:
                # Handle regular completion
                full_ids[rollout_idx].extend(new_tokens.flatten().tolist())
                full_attention_mask[rollout_idx].extend(torch.ones_like(new_tokens).flatten().tolist())
                generated_tokens_mask[rollout_idx].extend(torch.ones_like(new_tokens).flatten().tolist())
                current_prompts[rollout_idx] = current_prompts[rollout_idx] + completion_text
                

        # Update active rollouts
        active_rollouts = new_active_rollouts


    # Find max length across all rollouts
    max_length = max(len(ids) for ids in full_ids)
    
    # Pad sequences to max length
    for i in range(num_rollouts):
        # Pad full_ids with eos_token
        padding_length = max_length - len(full_ids[i])
        full_ids[i].extend([tokenizer.eos_token_id] * padding_length)
        
        # Pad attention and generated tokens masks with 0s
        full_attention_mask[i].extend([0] * padding_length)
        generated_tokens_mask[i].extend([0] * padding_length)
    
    # Convert lists to tensors
    full_ids = torch.tensor(full_ids, device=device)
    full_attention_mask = torch.tensor(full_attention_mask, device=device)
    generated_tokens_mask = torch.tensor(generated_tokens_mask, device=device)

   
    # Decode each sequence separately
    final_texts = [tokenizer.decode(ids[original_prompt_length:], skip_special_tokens=True) for ids in full_ids]
    

    # Process final results
    all_results = {
        'original_prompt_length': original_prompt_length,
        'full_ids': full_ids,
        'full_attention_mask': full_attention_mask,
        'generated_tokens_mask': generated_tokens_mask,
        'final_texts': final_texts,
        'tool_call_logs': tool_call_logs, 
        'prompt_text': prompt_text
    }


    return all_results

def _get_model_generations(args, arena, all_models, prompt_text, num_chains):
    """
    Generate responses from training and comparison models.
    
    Args:
        args: Generation configuration
        arena: Arena instance with tools
        all_models: Dictionary of model instances
        prompt_text: Input prompt
        num_chains: Number of parallel generations
        
    Returns:
        Tuple of:
        - Training model completions (list of strings)
        - Comparison model completions (list of strings)
        - Tool call logs for training model (list of lists of strings)
    """
    
    # Generate from training model
    training_generation_results = generate_completions_with_tools(
        args, arena, all_models["training_model"], 
        all_models["training_model_tokenizer"], prompt_text, 
        num_chains, device=all_models["training_model"].device
    )
    
    # Extract needed information from training results dictionary
    completions_text_for_scoring = training_generation_results['final_texts']
    tool_call_logs = training_generation_results['tool_call_logs']

    # Generate from comparison model
    compare_completions_text = []
    for _ in range(num_chains):
        completion = all_models["compare_model"].generate(
            system_prompt="You are an expert researcher.",
            user_prompt=prompt_text,
            temperature=args.temperature,
            max_new_tokens=args.max_completion_length
        )
        compare_completions_text.append(completion)
    
    return completions_text_for_scoring, compare_completions_text, tool_call_logs

def eval_on_test_set(all_models, arena, evaluator_instance, device, args, round_num):
    """
    Evaluate the model on the test set and generate PDF and JSON reports.
    
    Args:
        all_models: Dictionary containing all model instances
        arena: Arena instance with tools and questions
        evaluator_instance: Evaluator for computing rewards
        device: Computation device
        args: Training arguments
        round_num: Current training round number
        
    Returns:
        Tuple of:
        - Dictionary containing evaluation metrics
        - Final win rate percentage
    """
    total_scores = defaultdict(float)
    num_examples = 0
    total_comparisons_all = 0
    total_wins_all = 0
    total_tool_calls = 0

    # Setup output directories and files
    eval_log_dir = os.path.join(args.output_dir, 'eval_logs')
    os.makedirs(eval_log_dir, exist_ok=True)
    json_file_path = os.path.join(eval_log_dir, f'eval_results_round_{round_num}.json')
    pdf_file_path = os.path.join(eval_log_dir, f'eval_report_round_{round_num}.pdf')
    
    arena.reset_test_iterator()
    all_results_for_json = []
    
    # Evaluate each test question
    for question in arena.test_questions:
        # Format prompt with tool instructions
        tool_list_description = arena.tools.get_prompt_instructions()
        prompt_text = arena.generation_prompt.format(question=question, tool_list_description=tool_list_description)
        
        # Get model completions with tool call data
        completions_and_tools = _get_model_generations(
            args, arena, all_models, prompt_text, args.num_eval_chains
        )
        completions_text_for_scoring = completions_and_tools[0]
        compare_completions_text = completions_and_tools[1]
        tool_call_logs = completions_and_tools[2] if len(completions_and_tools) > 2 else None
        
        # Count tool calls for each completion
        completion_tool_calls = []
        if tool_call_logs:
            for logs in tool_call_logs:
                num_calls = len([log for log in logs if log.startswith("Tool call:")])
                completion_tool_calls.append(num_calls)
                total_tool_calls += num_calls
        else:
            # Count from text if logs not available
            for comp_text in completions_text_for_scoring:
                num_calls = len(utils.extract_tool_calls(comp_text))
                completion_tool_calls.append(num_calls)
                total_tool_calls += num_calls
        
        # Compute rewards and metrics
        rewards_per_func_all_comps, reward_metrics_for_question = evaluator_instance.compute_rewards(
            input_prompt=question, all_models=all_models, 
            train_model_completions=completions_text_for_scoring,
            compare_model_completions=compare_completions_text,
            device=device, is_test=True
        )

        # Process each comparison
        current_question_comparisons = []
        question_log_path = os.path.join(eval_log_dir, f'eval_log_round_{round_num}_question_{num_examples}.txt')
        
        for i, (trained_text, compare_text) in enumerate(zip(completions_text_for_scoring, compare_completions_text)):
            comparison_result = utils._process_single_comparison(
                trained_text, compare_text, question, arena, evaluator_instance, 
                all_models, rewards_per_func_all_comps[i], reward_metrics_for_question, i
            )
            
            # Add tool call count to the comparison result
            if i < len(completion_tool_calls):
                comparison_result["tool_calls_count"] = completion_tool_calls[i]
            
            current_question_comparisons.append(comparison_result)
            if comparison_result["winner"] == "trained": total_wins_all += 1
            total_comparisons_all += 1
        
        # Store results for logging and JSON
        question_results_for_log_and_json = {
            "question": question, 
            "comparisons": current_question_comparisons,
            "tool_calls": completion_tool_calls
        }
        utils._write_detailed_question_log(question_log_path, question, question_results_for_log_and_json)
        all_results_for_json.append(question_results_for_log_and_json)
        num_examples += 1

        # Aggregate scores for final metrics
        for comp_res in current_question_comparisons:
            for k, v in reward_metrics_for_question.items(): 
                if isinstance(v, (int, float)) and k.startswith('rewards/'):
                    total_scores[k] += v / len(current_question_comparisons)

        # Calculate win rate and average scores
        final_win_rate = (total_wins_all / total_comparisons_all) * 100 if total_comparisons_all > 0 else 0
        avg_scores_final = {k: v / num_examples for k, v in total_scores.items() if num_examples > 0}

    # Calculate average tool calls
    avg_tool_calls = total_tool_calls / total_comparisons_all if total_comparisons_all > 0 else 0

    # Compile final metrics
    final_metrics_summary = {
        'win_rate': final_win_rate, 
        'total_wins': total_wins_all,
        'total_comparisons': total_comparisons_all, 
        'num_examples': num_examples,
        'total_tool_calls': total_tool_calls,
        'avg_tool_calls': avg_tool_calls,
        'average_scores': avg_scores_final
    }

    # Save results to JSON and generate PDF report
    json_output_data = {
        "round": round_num, 
        "final_metrics": final_metrics_summary,
        "detailed_results": all_results_for_json
    }
    with open(json_file_path, 'w') as f_json: json.dump(json_output_data, f_json, indent=4)
    utils.generate_pdf_report(pdf_file_path, json_output_data)
    
    return final_metrics_summary, final_win_rate

def score_completions(completions_text, question, eval_class, all_models, device, args):
    """
    Score model completions and compute advantages for training.
    
    This function:
    1. Computes rewards for each completion using the evaluator
    2. Calculates advantages for policy gradient training
    3. Generates detailed logging data for analysis
    
    Args:
        completions_text: List of generated completion strings
        question: Original input question/prompt
        eval_class: Evaluator class for computing rewards
        all_models: Dictionary containing all models
        device: Device to place tensors on
        args: Training arguments
        
    Returns:
        Tuple containing:
        - rewards: Raw reward scores for each completion
        - advantages: Computed advantages for policy gradient
        - rewards_per_func: Rewards broken down by individual reward functions
        - metrics: Dictionary of aggregated metrics
        - log_data: Dictionary containing detailed generation and scoring data
    """
    # Initialize log data structure
    log_data = {
        'prompt': {
            'text': question,
        },
        'generations': []
    }

    # Compute rewards and metrics using evaluator
    rewards_per_func, metrics = eval_class.compute_rewards(
        input_prompt=question,
        all_models=all_models, 
        train_model_completions=completions_text, 
        compare_model_completions=None,
        device=device, 
        is_test=False
    )
    rewards = rewards_per_func.sum(dim=1)

    # Store generation data with scores
    for i, (completion, reward_scores) in enumerate(zip(completions_text, rewards_per_func)):
        generation_data = {
            'response': completion,
            'scores': {
                **eval_class.get_reward_breakdown(reward_scores),
                'total_reward': rewards[i].item()
            }
        }
        log_data['generations'].append(generation_data)

    # Compute advantages for policy gradient
    # Group rewards by number of chains and compute statistics
    mean_grouped_rewards = rewards.view(-1, args.num_chains).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, args.num_chains).std(dim=1)

    # Expand statistics to match original reward dimensions
    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(args.num_chains, dim=0)

    # Calculate normalized advantages
    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    metrics["reward_std"] = std_grouped_rewards.mean().item()

    # Store summary statistics for logging
    log_data['summary_stats'] = {
        'mean_rewards_per_group': mean_grouped_rewards.tolist(),
        'std_rewards_per_group': std_grouped_rewards.tolist(),
        'advantages': advantages.tolist()
    }

    return rewards, advantages, rewards_per_func, metrics, log_data

def compute_loss(model: PreTrainedModel, 
                 prompt_ids, 
                 completion_ids, 
                 attention_mask, 
                 loss_mask, 
                 advantages, 
                 args):
    """
    Compute policy gradient loss.
    
    This function:
    1. Calculates per-token log probabilities for completions
    2. Applies policy gradient loss with advantages
    3. Handles masking for proper loss computation on generated tokens
    
    Args:
        model: The current model being trained
        prompt_ids: Token IDs for the prompt part (B, P_L)
        completion_ids: Token IDs for the completion part (B, C_L)
        attention_mask: Attention mask for the full sequence (prompt + completion) (B, P_L + C_L)
        loss_mask: Mask indicating which completion tokens should contribute to loss (B, C_L)
        advantages: Advantage values for each sequence (B,)
        args: Training arguments
        
    Returns:
        Tuple containing:
        - loss: The computed policy gradient loss
        - metrics: Dictionary containing additional metrics
    """
    # Concatenate prompt and completion for model input
    input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    
    logits_to_keep = completion_ids.size(1) 
    
    # Get training model log probabilities for the completion part
    per_token_logps = utils.get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

    # Compute loss with advantages (policy gradient)
    per_token_loss = -torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    
    # Align loss_mask with completion_attention_mask
    completion_attention_mask = attention_mask[:, prompt_ids.size(1):]
    effective_loss_mask = loss_mask * completion_attention_mask
    assert torch.all(effective_loss_mask == loss_mask), "effective_loss_mask should equal loss_mask"
    # Calculate percentage of tokens that are masked out
    loss_mask_masked = (loss_mask == 0).sum().item()
    completion_mask_masked = (completion_attention_mask == 0).sum().item()


    # Compute loss with proper normalization across all tokens
    loss = (per_token_loss * effective_loss_mask).sum() / (per_token_loss.size(0) * args.max_completion_length)

    # Calculate additional metrics
    metrics = {
        "response_length": effective_loss_mask.sum(1).float().mean().item()
    }

    return loss, metrics

def grpo_loss(arena, all_models, question, eval_class, device, round_num, training_log_dir, args):
    """
    Compute policy gradient loss for the training model.
    
    This function:
    1. Generates completions with tool interactions
    2. Scores completions and computes advantages
    3. Generates training reports and logs
    4. Computes the final policy gradient loss
    
    Args:
        arena: The arena instance with tools
        all_models: Dictionary containing all models
        question: Input question/prompt
        eval_class: Evaluator for computing rewards
        device: Device to run on ('cpu' or 'cuda')
        round_num: Current training round number
        training_log_dir: Directory to save training logs
        args: Training arguments
        
    Returns:
        Tuple containing:
        - loss: The computed policy gradient loss
        - metrics: Dictionary containing training metrics
    """

    ############################
    ### Generate Completions ###
    ############################
    model_device = all_models["training_model"].device # Get the actual device of the training model
    tool_list_description = arena.tools.get_prompt_instructions()
    prompt_text = arena.generation_prompt.format(question=question, tool_list_description=tool_list_description)

    # Generate completions results (dictionary format)
    generation_results = generate_completions_with_tools(
        args, arena, all_models["training_model"], 
        all_models["training_model_tokenizer"], prompt_text, 
        args.num_chains, device=model_device # Ensure generation happens on model's device
    )
    
    # Extract necessary data from results
    completions_text = generation_results['final_texts']
    tool_call_logs = generation_results['tool_call_logs']
    full_ids = generation_results['full_ids'] # Shape: [batch_size, seq_len]
    full_attention_mask = generation_results['full_attention_mask'] # Shape: [batch_size, seq_len]
    # generated_tokens_mask is the loss_mask for the completion part, it excludes prompt and tool responses.
    # Shape: [batch_size, seq_len], but we will slice it for completion part. 
    original_prompt_length = generation_results['original_prompt_length']

    ##################################
    ### Score + Comptute Advantage ###
    ##################################
    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text, question, eval_class, all_models, model_device, args
    )
    

    ####################
    ### Compute Loss ###
    ####################
    # Prepare inputs for the new compute_loss structure
    prompt_ids_batched = full_ids[:, :original_prompt_length]
    completion_ids_batched = full_ids[:, original_prompt_length:]
    
    # The generated_tokens_mask from generate_completions_with_tools is for the *entire* sequence (prompt + completion).
    # The prompt part is already 0s. We need the part corresponding to completion_ids_batched.
    loss_mask_for_completion = generation_results['generated_tokens_mask'][:, original_prompt_length:]

    loss, batch_metrics = compute_loss(
        all_models["training_model"],
        prompt_ids_batched, 
        completion_ids_batched, 
        full_attention_mask, 
        loss_mask_for_completion, 
        advantages, 
        args
    )
    
    ##########################
    ### Logging + Clean Up ###
    #########################
    
    # Calculate tool call metrics
    total_tool_calls = sum(len([log for log in logs if log.startswith("Tool call:")]) for logs in tool_call_logs)
    avg_tool_calls = total_tool_calls / len(completions_text) if completions_text else 0
    
   # Add tool call metrics to overall metrics
    metrics["total_tool_calls"] = total_tool_calls
    metrics["avg_tool_calls"] = avg_tool_calls
    
    # Get round robin tournament results if available
    round_robin_results = eval_class.get_round_robin_results()
    
    # Generate training report
    # pdf_path = os.path.join(training_log_dir, f'training_report_round_{round_num}.pdf')
    # utils.generate_training_pdf_report(
    #     pdf_path, 
    #     question, 
    #     completions_text, 
    #     rewards_per_func, 
    #     advantages, 
    #     eval_class,
    #     round_robin_results,
    #     tool_call_logs
    # )

    # Write generation log
    log_file = os.path.join(training_log_dir, f'{round_num}_generations.txt')
    utils.write_generation_log(log_data, log_file)

    # Update metrics with batch metrics
    metrics.update(batch_metrics)
    
    # Clean up large tensors to manage GPU memory
    del full_ids, full_attention_mask # Removed generated_tokens_mask as it's sliced into loss_mask_for_completion
    del advantages, rewards, rewards_per_func, generation_results, loss_mask_for_completion, prompt_ids_batched, completion_ids_batched
    
    return loss, metrics

def parse_args():
    """
    Parse command line arguments for training configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments containing:
        - Model configuration (model names, sizes)
        - Environment settings (arena, tools)
        - Output and logging options
        - Optimization hyperparameters
        - Generation parameters
        - Training parameters
    """
    parser = argparse.ArgumentParser(description="pico Deep Research trainer")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B", help="Name/path of base model to train")
    parser.add_argument("--judge_model_name", type=str, default="gpt-4o-mini", help="Model to use for reward computation and evaluation")
    parser.add_argument("--compare_model_name", type=str, default="gpt-4o-mini", help="Model to use for performance comparison")

    # Environment configuration
    parser.add_argument("--arena_name", type=str, default="debate", choices=["debate"], help="Type of training arena to use")
    parser.add_argument("--tool_registry", type=str, default="websearch", choices=["websearch"], help="Set of tools available to the model")

    # Output and logging configuration
    parser.add_argument("--output_dir", type=str, default="output", help="Directory for saving outputs and checkpoints")
    parser.add_argument("--verbose", action="store_true", help="Enable detailed logging")
    parser.add_argument("--save_steps", type=int, default=80, help="Save model checkpoint every N steps")
    parser.add_argument("--eval_iterations", type=int, default=40, help="Run evaluation every N iterations")

    # Optimization hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate for Adam optimizer")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Beta1 parameter for Adam optimizer")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Beta2 parameter for Adam optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for regularization")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Maximum gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps to accumulate gradients")
    parser.add_argument("--warmup_percent", type=float, default=0.18, help="Percentage of total steps for learning rate warmup")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature for generation")
    parser.add_argument("--num_chains", type=int, default=10,  help="Number of parallel generation chains during training")
    parser.add_argument("--num_eval_chains", type=int, default=6, help="Number of parallel generation chains during evaluation")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum length of input prompts")
    parser.add_argument("--max_completion_length", type=int, default=1512, help="Maximum length of generated completions")
    parser.add_argument("--max_tool_interactions", type=int, default=3, help="Maximum number of tool calls per generation")
    parser.add_argument("--generation_chunk_size", type=int, default=128, help="Number of tokens to generate per step during tool interaction")

    # Training parameters
    parser.add_argument("--num_train_iters", type=int, default=1000, help="Total number of training iterations")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--seed", type=int, default=7111994, help="Random seed for reproducibility")

    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args() 
    utils.seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    # Initialize models
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)  # Model to be trained
    judge_model = llms.get_judge_model(args.judge_model_name, device)  # Model for reward computation
    compare_model = llms.get_compare_model(args.compare_model_name, device)  # Model for performance comparison
    all_models = {
        "training_model": model,
        "training_model_tokenizer": tokenizer,
        "judge_model": judge_model,
        "compare_model": compare_model
    }

    # Initialize training environment
    tool_registry = tools.create_tool_registry(args.tool_registry)
    arenastage_instance = arenastages.get_arena_stage(args.arena_name)
    arena_instance = arenas.Arena(arenastage_instance, tool_registry)
    evaluator_instance = evaluator.GenericEvaluator(arenastage_instance)

    # Setup output directories
    os.makedirs(args.output_dir, exist_ok=True)
    args_path = os.path.join(args.output_dir, 'args.json')
    with open(args_path, 'w') as f_json_args:
        json.dump(vars(args), f_json_args, indent=4)
    
    eval_log_dir = os.path.join(args.output_dir, 'eval_logs')
    os.makedirs(eval_log_dir, exist_ok=True)
    training_log_dir = os.path.join(args.output_dir, 'training_logs')
    os.makedirs(training_log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr_lambda(step):
        if step < warmup_steps: 
            return float(step) / float(max(1, warmup_steps))
        return 1.0  # Constant learning rate after warmup
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda)
    
    # Initialize training state
    start_round = 0
    train_metrics_total = {}

    # Resume training from checkpoint if requested
    if args.resume:
        checkpoint_path = os.path.join(checkpoint_dir, f'latest_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            gc.collect()
            torch.cuda.empty_cache()
            try:
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                optimizer.zero_grad()  # Reset gradients after loading states
                start_round = checkpoint.get('round_num', 0) + 1
                train_metrics_total = checkpoint.get('train_metrics_total', {})
            except Exception as e:
                import traceback
    # Main training loop
    for round_num in tqdm(range(start_round, args.num_train_iters), desc="Training Progress"):
        # Periodic evaluation
        if round_num % args.eval_iterations == 0 and round_num > 0: 
            with torch.no_grad():
                eval_summary, _ = eval_on_test_set(
                    all_models, arena_instance, evaluator_instance, device, args, round_num
                )

        # Periodic checkpointing
        if (round_num + 1) % args.save_steps == 0:
            save_path = os.path.join(checkpoint_dir, f'checkpoint_round_{round_num}.pt')
            torch.save({
                'round_num': round_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics_total': train_metrics_total,
                'args': args
            }, save_path)
            # Save latest checkpoint for easy resume
            latest_save_path = os.path.join(checkpoint_dir, f'latest_checkpoint.pt')
            torch.save({
                'round_num': round_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics_total': train_metrics_total,
                'args': args
            }, latest_save_path)

        # Training step
        question = arena_instance.get_train_example()
        loss, train_step_metrics = grpo_loss(
            arena_instance, all_models, question, evaluator_instance, 
            device, round_num, training_log_dir, args
        )

        # Gradient accumulation and optimization
        loss_for_backward = loss #/ args.gradient_accumulation_steps
        loss_for_backward.backward()
        
        # Calculate and log gradient norm before clipping
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                grad_norm += param_norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        train_step_metrics["grad_norm"] = grad_norm
        
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
        
        scheduler.step()  # Update learning rate

        # Logging and metrics tracking
        current_lr = scheduler.get_last_lr()[0]
        train_step_metrics["learning_rate"] = current_lr
        train_step_metrics["loss_raw"] = loss.item()
        
        # Store metrics
        train_metrics_total[str(round_num)] = train_step_metrics
        
        # Save individual round metrics
        round_metrics_file = os.path.join(training_log_dir, f'metrics_round_{round_num}.json')
        with open(round_metrics_file, 'w') as f_round_metrics:
            json.dump(train_step_metrics, f_round_metrics, indent=4)
        
        # Update aggregated metrics periodically
        with open(os.path.join(training_log_dir, "train_metrics_log.json"), "w") as f_train_log:
            json.dump(train_metrics_total, f_train_log, indent=4)
        

        # Memory management
        del loss, train_step_metrics, loss_for_backward
        gc.collect()
        torch.cuda.empty_cache()

    # Save final model
    final_save_path = os.path.join(checkpoint_dir, f'final_model_round_{args.num_train_iters-1}.pt')
    torch.save({
        'round_num': args.num_train_iters - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_metrics_total': train_metrics_total,
        'args': args
    }, final_save_path)
    