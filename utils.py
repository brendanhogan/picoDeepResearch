"""
Utility functions for model training and evaluation.

Provides:
- Text processing and cleaning
- Random seed management
- Generation logging
- Token probability computation
"""

import os
import torch
import random
import numpy as np
import torch.nn.functional as F
from typing import Any, Dict, Optional
import re

####################
## MISC FUNCTIONS ##
####################

def clean_spaces_preserve_newlines(text: str) -> str:
    """
    Clean text by removing extra spaces while preserving newlines.
    
    Args:
        text: Input text to clean
        
    Returns:
        Text with normalized spacing and preserved newlines
    """
    lines = text.split("\n")  # Split by newlines
    cleaned_lines = [" ".join(re.split(r"\s+", line)).strip() for line in lines]  # Remove extra spaces in each line
    return "\n".join(cleaned_lines)  # Join the lines back with newlines



def seed_everything(seed: int) -> None:
    """
    Set consistent random seeds across all libraries.
    
    Ensures reproducible results by configuring:
    - Python random module
    - NumPy random state
    - PyTorch random state (CPU and CUDA)
    - CUDNN deterministic mode
    
    Args:
        seed: Random seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Configure CUDNN for deterministic operation
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def write_generation_log(log_data: Dict[str, Any], log_file: str) -> None:
    """
    Write model generation results to a structured log file.
    
    Log format:
    - Original prompt
    - Generated responses
    - Parsed XML sections (reasoning/answer)
    - Reward scores and total
    
    Args:
        log_data: Dictionary containing generation data
        log_file: Output file path
    """
    with open(log_file, 'w') as f:
        # Write prompt section
        f.write("###### ORIGINAL PROMPT #####\n\n")
        f.write(log_data['prompt']['text'] + "\n\n")

        # Write each generation
        for i, gen in enumerate(log_data['generations'], 1):
            f.write(f"#### GENERATION {i} ####\n\n")
            f.write("RESPONSE:\n")
            f.write(gen['response'] + "\n\n")
            
            # Parse XML sections if present
            try:
                reasoning = gen['response'].split("<reasoning>\n")[1].split("\n</reasoning>")[0]
                answer = gen['response'].split("<answer>\n")[1].split("\n</answer>")[0]
                f.write("PARSED SECTIONS:\n")
                f.write(f"Reasoning:\n{reasoning}\n")
                f.write(f"Answer:\n{answer}\n\n")
            except:
                f.write("ERROR: Could not parse XML sections\n\n")
            
            # Write reward scores
            f.write("REWARD SCORES:\n")
            for reward_name, reward_value in gen['scores'].items():
                f.write(f"{reward_name}: {reward_value:.4f}\n")
            # Total reward is sum of individual scores
            total_reward = sum(gen['scores'].values())
            f.write(f"Total reward: {total_reward:.4f}\n\n")
            f.write("-"*40 + "\n\n")


####################################################################################
## Copied Directly from TRL -> generate log probs per token                 ########
## https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py ########
####################################################################################

def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Memory-efficient implementation of log_softmax with selective gathering.
    
    Optimized version of:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```
    
    Args:
        logits: Logits tensor of shape (..., num_classes)
        index: Index tensor of shape (...)
        
    Returns:
        Gathered log probabilities with same shape as index
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # Compute logsumexp in chunks to reduce memory usage
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # Handle bfloat16 with more stable approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # Process in chunks to reduce memory
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps

def get_per_token_logps(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
    """
    Compute log probabilities for each token in the sequence.
    
    Args:
        model: Language model to use
        input_ids: Input token IDs
        attention_mask: Attention mask
        logits_to_keep: Number of logits to keep from sequence end
        
    Returns:
        Log probabilities for each token
    """
    # Get logits for sequence plus one token
    logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
    logits = logits[:, :-1, :]  # Remove last logit (next token prediction)

    # Keep only specified number of logits from end
    input_ids = input_ids[:, -logits_to_keep:]
    logits = logits[:, -logits_to_keep:]
    
    return selective_log_softmax(logits, input_ids)  # Compute log probabilities



## Othe helper/plotter functions

def extract_section(text: str, tag: str) -> list:
    """
    Extract content between XML tags.
    
    Args:
        text: Input text containing XML
        tag: Tag name to extract
        
    Returns:
        List of extracted content strings
    """
    import re
    pattern = f'<{tag}>(.*?)</{tag}>'
    matches = re.findall(pattern, text, re.DOTALL)
    return matches if matches else []

def extract_tool_calls(text: str) -> list:
    """
    Extract all tool calls from text.
    
    Args:
        text: Input text containing tool calls
        
    Returns:
        List of tool call XML strings
    """
    import re
    pattern = r'<tool.*?</tool>'
    return re.findall(pattern, text, re.DOTALL)





def _process_single_comparison(trained_model_text, comp_text, question, arena, evaluator_instance, all_models, rewards_per_func_i, reward_metrics_for_question, i):
    """Processes results for a single comparison between trained and compare model."""
    think_parts = extract_section(trained_model_text, 'think')
    answer_parts = extract_section(trained_model_text, 'answer')
    tool_calls = extract_tool_calls(trained_model_text)

    trained_score_this_comp = 0
    compare_score_this_comp = 0
    overall_winner_this_comp = "tie"

    if "principle_judgements" in reward_metrics_for_question and \
       reward_metrics_for_question["principle_judgements"] and \
       i < len(reward_metrics_for_question["principle_judgements"]):
        
        judgements_for_current_comp = reward_metrics_for_question["principle_judgements"][i]
        for pj_detail in judgements_for_current_comp:
            weight = pj_detail.get("weight", 0)
            winner_code = pj_detail.get("winner_code", 0)
            if winner_code == 1: trained_score_this_comp += weight
            elif winner_code == 2: compare_score_this_comp += weight
            elif winner_code == 0: 
                trained_score_this_comp += weight / 2
                compare_score_this_comp += weight / 2
        
        if trained_score_this_comp > compare_score_this_comp: overall_winner_this_comp = "trained"
        elif compare_score_this_comp > trained_score_this_comp: overall_winner_this_comp = "compare"

    return {
        "trained_model_text": trained_model_text,
        "compare_model_text": comp_text,
        "rewards_breakdown": evaluator_instance.get_reward_breakdown(rewards_per_func_i),
        "winner": overall_winner_this_comp,
        "sections": {"think": think_parts, "answer": answer_parts, "tool_calls": tool_calls},
        "judging_details": {
            "raw_rewards_tensor": rewards_per_func_i.tolist(),
            "metrics_from_evaluator": reward_metrics_for_question, # Contains all principle_judgements for the *question*
            "principle_scores_this_comparison": {
                "trained_weighted_score": trained_score_this_comp,
                "compare_weighted_score": compare_score_this_comp
            }
        }
    }


def _write_detailed_question_log(question_log_path, question, question_results):
    """Writes the detailed log for a single evaluated question."""
    with open(question_log_path, 'w', encoding='utf-8') as f_log:
        f_log.write(f"Question: {question}\n\n")
        for i, comparison in enumerate(question_results["comparisons"]):
            f_log.write(f"\nComparison #{i+1}:\n{'-'*40}\n")
            f_log.write(f"TRAINED MODEL:\n{comparison['trained_model_text']}\n\n")
            f_log.write(f"COMPARE MODEL:\n{comparison['compare_model_text']}\n\n")
            f_log.write("JUDGING DETAILS:\n")
            
            current_comparison_idx_for_log = i # Direct index
            judging_metrics = comparison["judging_details"]["metrics_from_evaluator"]
            
            if "principle_judgements" in judging_metrics and \
               judging_metrics["principle_judgements"] and \
               current_comparison_idx_for_log < len(judging_metrics["principle_judgements"]):
                judgements_for_this_log_entry = judging_metrics["principle_judgements"][current_comparison_idx_for_log]
                if not judgements_for_this_log_entry:
                    f_log.write("  No principle judgements found for this specific comparison.\n")
                else:
                    for eval_detail in judgements_for_this_log_entry:
                        f_log.write(f"\n  Principle: {eval_detail.get('principle', 'N/A')}\n")
                        f_log.write(f"  Weight: {eval_detail.get('weight', 'N/A')}\n")
                        winner_map = {1: "TRAINED MODEL", 2: "COMPARE MODEL", 0: "TIE"}
                        f_log.write(f"  Winner: {winner_map.get(eval_detail.get('winner_code', -1), 'UNKNOWN')}\n")
                        f_log.write(f"  Judge Explanation: {eval_detail.get('explanation', 'No explanation provided.')}\n")
            else:
                f_log.write("  No principle judgements data found or not in expected format.\n")

            scores_this_comp = comparison["judging_details"].get("principle_scores_this_comparison", {})
            f_log.write(f"\n  Scores for this comparison (based on principles):\n")
            f_log.write(f"    Trained Model Weighted Score: {scores_this_comp.get('trained_weighted_score', 0):.2f}\n")
            f_log.write(f"    Compare Model Weighted Score: {scores_this_comp.get('compare_weighted_score', 0):.2f}\n")
            f_log.write(f"\nOVERALL WINNER (this comparison): {comparison['winner'].upper()}\n")

            if comparison["sections"]["think"]: f_log.write("\nTHINKING PROCESS:\n" + "\n\n".join(comparison["sections"]["think"]) + "\n\n")
            if comparison["sections"]["answer"]: f_log.write("\nFINAL ANSWER:\n" + "\n\n".join(comparison["sections"]["answer"]) + "\n\n")
            if comparison["sections"]["tool_calls"]: f_log.write("\nTOOL CALLS:\n" + "\n\n".join(comparison["sections"]["tool_calls"]) + "\n\n")



def generate_pdf_report(pdf_path: str, results: dict):
    """Generate a PDF report from evaluation results."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    import re
    
    def escape_tags(text):
        # Replace < and > with their HTML entities
        return text.replace('<', '&lt;').replace('>', '&gt;')
    
    def extract_section(text, tag):
        # Extract content between specific tags
        pattern = f'<{tag}>(.*?)</{tag}>'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches if matches else []
    
    def extract_tool_calls(text):
        # Extract all tool calls
        pattern = r'<tool.*?</tool>'
        return re.findall(pattern, text, re.DOTALL)
    
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Create styles for different sections
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading4'],
        fontSize=12,
        textColor=colors.darkblue,
        spaceAfter=6
    )
    
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph(f"Evaluation Report - Round {results['round']}", title_style))
    
    # Summary metrics
    story.append(Paragraph("Summary Metrics", styles['Heading2']))
    metrics = results['final_metrics']
    metrics_data = [
        ["Metric", "Value"],
        ["Win Rate", f"{metrics['win_rate']:.2f}%"],
        ["Total Wins", str(metrics['total_wins'])],
        ["Total Comparisons", str(metrics['total_comparisons'])],
        ["Number of Examples", str(metrics['num_examples'])],
        ["Total Tool Calls", str(metrics.get('total_tool_calls', 0))],
        ["Average Tool Calls per Completion", f"{metrics.get('avg_tool_calls', 0):.2f}"]
    ]
    
    metrics_table = Table(metrics_data, colWidths=[200, 200])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Detailed results
    story.append(Paragraph("Detailed Results", styles['Heading2']))
    for q_idx, question_result in enumerate(results['detailed_results'], 1):
        story.append(Paragraph(f"Question {q_idx}", styles['Heading3']))
        story.append(Paragraph(escape_tags(question_result['question']), styles['Normal']))
        story.append(Spacer(1, 10))
        
        for c_idx, comparison in enumerate(question_result['comparisons'], 1):
            # Get tool call count if available
            tool_count = comparison.get("tool_calls_count", 0)
            if not tool_count and "tool_calls" in question_result and c_idx-1 < len(question_result["tool_calls"]):
                tool_count = question_result["tool_calls"][c_idx-1]
            
            story.append(Paragraph(f"Comparison {c_idx} (Tool Calls: {tool_count})", styles['Heading4']))
            
            # Full response
            story.append(Paragraph("Full Response:", section_style))
            story.append(Paragraph(escape_tags(comparison['trained_model_text']), styles['Normal']))
            story.append(Spacer(1, 10))
            
            # Extract and display sections
            think_parts = extract_section(comparison['trained_model_text'], 'think')
            answer_parts = extract_section(comparison['trained_model_text'], 'answer')
            tool_calls = extract_tool_calls(comparison['trained_model_text'])
            
            if think_parts:
                story.append(Paragraph("Thinking Process:", section_style))
                for think in think_parts:
                    story.append(Paragraph(escape_tags(think), styles['Normal']))
                story.append(Spacer(1, 10))
            
            if answer_parts:
                story.append(Paragraph("Final Answer:", section_style))
                for answer in answer_parts:
                    story.append(Paragraph(escape_tags(answer), styles['Normal']))
                story.append(Spacer(1, 10))
            
            if tool_calls:
                story.append(Paragraph(f"Tool Calls ({len(tool_calls)}):", section_style))
                for tool in tool_calls:
                    story.append(Paragraph(escape_tags(tool), styles['Normal']))
                story.append(Spacer(1, 10))
            
            # Compare model
            story.append(Paragraph("Compare Model:", section_style))
            story.append(Paragraph(escape_tags(comparison['compare_model_text']), styles['Normal']))
            story.append(Spacer(1, 10))
            
            # Judging Details
            story.append(Paragraph("Judging Details:", section_style))
            
            # Display principle evaluations with explanations
            story.append(Paragraph("<b>Principle Evaluations:</b>", styles['Heading5'])) # Bolded header
            story.append(Spacer(1, 6))
            
            current_comparison_judging_details = comparison.get('judging_details', {})
            current_comparison_metrics = current_comparison_judging_details.get('metrics_from_evaluator', {})
            
            # logger.info(f"PDF: Q{q_idx} C{c_idx}, Keys in current_comparison_metrics: {current_comparison_metrics.keys()}")


            if "principle_judgements" in current_comparison_metrics and \
               current_comparison_metrics["principle_judgements"] and \
               c_idx -1 < len(current_comparison_metrics["principle_judgements"]): # c_idx is 1-based for PDF loop

                judgements_for_this_pdf_comparison = current_comparison_metrics["principle_judgements"][c_idx - 1]
                # logger.info(f"PDF: Q{q_idx} C{c_idx}, Found {len(judgements_for_this_pdf_comparison)} judgements.")

                if not judgements_for_this_pdf_comparison:
                    story.append(Paragraph("<i>No principle judgements found for this specific comparison.</i>", styles['Normal']))
                else:
                    for eval_detail in judgements_for_this_pdf_comparison:
                        story.append(Paragraph(f"<b>Principle:</b> {escape_tags(eval_detail.get('principle', 'N/A'))}", styles['Normal']))
                        story.append(Paragraph(f"<b>Weight:</b> {eval_detail.get('weight', 'N/A')}", styles['Normal']))
                        winner_map = {1: "TRAINED MODEL", 2: "COMPARE MODEL", 0: "TIE"}
                        story.append(Paragraph(f"<b>Winner:</b> {winner_map.get(eval_detail.get('winner_code', -1), 'UNKNOWN')}", styles['Normal']))
                        story.append(Paragraph(f"<b>Judge Explanation:</b>", styles['Normal'])) # Label
                        # Use a slightly indented style for the explanation itself for better readability
                        explanation_style = ParagraphStyle('ExplanationStyle', parent=styles['Normal'], leftIndent=18)
                        story.append(Paragraph(escape_tags(eval_detail.get('explanation', 'No explanation provided.')), explanation_style))
                        story.append(Spacer(1, 8)) # More space after each principle block
            else:
                story.append(Paragraph(f"<i>Principle judgements data not found or not in expected format for comparison {c_idx}.</i>", styles['Normal']))
                # logger.info(f"PDF: Q{q_idx} C{c_idx}, 'principle_judgements' key missing or list empty/too short. Metrics keys: {current_comparison_metrics.keys()}")


            # Display principle scores for this comparison
            scores_this_comp = current_comparison_judging_details.get("principle_scores_this_comparison", {})
            story.append(Spacer(1,6))
            story.append(Paragraph(f"<b>Scores for this comparison (based on principles):</b>", styles['Normal']))
            story.append(Paragraph(f"  Trained Model Weighted Score: {scores_this_comp.get('trained_weighted_score', 0):.2f}", styles['Normal']))
            story.append(Paragraph(f"  Compare Model Weighted Score: {scores_this_comp.get('compare_weighted_score', 0):.2f}", styles['Normal']))
            story.append(Spacer(1,10))

            # Display other metrics from evaluator (if any, besides principle judgements)
            # story.append(Paragraph("<b>Other Raw Metrics from Evaluator:</b>", styles['Heading5']))
            # for metric_name, metric_value in current_comparison_metrics.items():
            #     if metric_name != 'principle_judgements': # Avoid re-printing
            #         story.append(Paragraph(f"{metric_name}: {str(metric_value)}", styles['Normal']))
            
            # Overall winner
            story.append(Paragraph(f"<b>Overall Winner (this comparison):</b> {comparison.get('winner', 'N/A').upper()}", section_style))
            story.append(Spacer(1, 20))
    
    doc.build(story)


def generate_training_pdf_report(pdf_path, question, completions_text, rewards_per_func, advantages, eval_class, round_robin_results=None, tool_call_logs=None):
    """Generate a PDF report for training data with detailed completion information and tournament results."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    import re
    
    def escape_tags(text):
        # Replace < and > with their HTML entities
        return text.replace('<', '&lt;').replace('>', '&gt;')
    
    def extract_section(text, tag):
        # Extract content between specific tags
        pattern = f'<{tag}>(.*?)</{tag}>'
        matches = re.findall(pattern, text, re.DOTALL)
        return matches if matches else []
    
    def extract_tool_calls(text):
        # Extract all tool calls
        pattern = r'<tool.*?</tool>'
        return re.findall(pattern, text, re.DOTALL)
    
    def count_tool_calls(text_or_logs):
        """Count the number of tool calls either from text or from logs"""
        if isinstance(text_or_logs, list):
            # Count from logs (every even entry is a tool call)
            return len([log for log in text_or_logs if log.startswith("Tool call:")])
        else:
            # Count from text
            return len(extract_tool_calls(text_or_logs))
    
    # Create a landscape-oriented document
    page_width, page_height = landscape(letter)
    doc = SimpleDocTemplate(pdf_path, pagesize=landscape(letter), leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    # Create styles for different sections
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=12
    )
    
    subheading_style = ParagraphStyle(
        'Subheading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8
    )
    
    section_style = ParagraphStyle(
        'Section',
        parent=styles['Heading4'],
        fontSize=12,
        textColor=colors.darkblue,
        spaceAfter=6
    )
    
    normal_style = styles['Normal']
    
    code_style = ParagraphStyle(
        'Code',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=9,
        leftIndent=20
    )
    
    # Style for table cells with wrapped text
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,  # Slightly tighter line spacing
        spaceBefore=0,
        spaceAfter=0
    )
    
    story = []
    
    # Title
    story.append(Paragraph(f"Training Report", title_style))
    
    # Question
    story.append(Paragraph("Question:", heading_style))
    story.append(Paragraph(escape_tags(question), normal_style))
    story.append(Spacer(1, 20))
    
    # Completions
    story.append(Paragraph("Completions:", heading_style))
    
    # Dictionary to store extracted parts for ranking
    extracted_data = []
    
    # Track tool calls for statistics
    total_tool_calls = 0
    
    # Process each completion
    for i, (completion, rewards) in enumerate(zip(completions_text, rewards_per_func)):
        story.append(Paragraph(f"Completion #{i+1}", subheading_style))
        
        # Extract sections
        think_parts = extract_section(completion, 'think')
        answer_parts = extract_section(completion, 'answer')
        tool_calls_text = extract_tool_calls(completion)
        
        # Count tool calls
        num_tool_calls = 0
        if tool_call_logs and i < len(tool_call_logs):
            num_tool_calls = count_tool_calls(tool_call_logs[i])
        else:
            num_tool_calls = len(tool_calls_text)
        
        total_tool_calls += num_tool_calls
        
        # Store for ranking later
        extracted_data.append({
            'id': i+1,
            'think': '\n'.join(think_parts) if think_parts else '',
            'answer': '\n'.join(answer_parts) if answer_parts else '',
            'tool_calls': tool_calls_text,
            'num_tool_calls': num_tool_calls,
            'total_reward': rewards.sum().item(),
            'rewards': eval_class.get_reward_breakdown(rewards)
        })
        
        # Full text
        story.append(Paragraph("Full Text:", section_style))
        story.append(Paragraph(escape_tags(completion), code_style))
        story.append(Spacer(1, 10))
        
        # Thinking
        if think_parts:
            story.append(Paragraph("Thinking Process:", section_style))
            for think in think_parts:
                story.append(Paragraph(escape_tags(think), code_style))
            story.append(Spacer(1, 10))
        
        # Tool calls
        if tool_calls_text:
            story.append(Paragraph(f"Tool Calls ({num_tool_calls}):", section_style))
            for tool in tool_calls_text:
                story.append(Paragraph(escape_tags(tool), code_style))
            story.append(Spacer(1, 10))
        
        # Answer
        if answer_parts:
            story.append(Paragraph("Answer:", section_style))
            for answer in answer_parts:
                story.append(Paragraph(escape_tags(answer), code_style))
            story.append(Spacer(1, 10))
        
        # Scores
        story.append(Paragraph("Scores:", section_style))
        reward_breakdown = eval_class.get_reward_breakdown(rewards)
        
        scores_data = [["Metric", "Value"]]
        for metric, value in reward_breakdown.items():
            scores_data.append([metric, f"{value:.4f}"])
        scores_data.append(["Total Reward", f"{rewards.sum().item():.4f}"])
        scores_data.append(["Tool Calls", str(num_tool_calls)])
        
        scores_table = Table(scores_data, colWidths=[200, 100])
        scores_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(scores_table)
        story.append(Spacer(1, 20))
    
    # Tool call statistics
    avg_tool_calls = total_tool_calls / len(completions_text) if completions_text else 0
    
    # Add tool call summary section
    story.append(Paragraph("Tool Call Statistics:", heading_style))
    tool_stats_data = [
        ["Metric", "Value"],
        ["Total Tool Calls", str(total_tool_calls)],
        ["Average Tool Calls per Completion", f"{avg_tool_calls:.2f}"]
    ]
    
    tool_stats_table = Table(tool_stats_data, colWidths=[200, 100])
    tool_stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(tool_stats_table)
    story.append(Spacer(1, 20))
    
    # Ranking Table
    story.append(Paragraph("Completions Ranked by Total Reward:", heading_style))
    
    # Sort data by total_reward
    ranked_data = sorted(extracted_data, key=lambda x: x['total_reward'], reverse=True)
    
    # Calculate available width for the table (page width minus margins)
    available_width = page_width - 72  # 36 points margin on each side
    
    # Set column widths with appropriate proportions
    rank_col_width = int(available_width * 0.06)      # 6%
    id_col_width = int(available_width * 0.10)        # 10%
    reward_col_width = int(available_width * 0.12)    # 12%
    tools_col_width = int(available_width * 0.12)     # 12%
    answer_col_width = int(available_width * 0.60)    # 60%
    
    # Create ranking table with Paragraph objects for text wrapping
    ranking_data = [["Rank", "Completion ID", "Total Reward", "Tool Calls", "Answer (truncated)"]]
    for rank, item in enumerate(ranked_data, 1):
        # Truncate answer for display and wrap in a Paragraph
        truncated_answer = item['answer'][:300] + "..." if len(item['answer']) > 300 else item['answer']
        answer_paragraph = Paragraph(escape_tags(truncated_answer), table_cell_style)
        
        ranking_data.append([
            str(rank), 
            f"#{item['id']}", 
            f"{item['total_reward']:.4f}",
            str(item['num_tool_calls']),
            answer_paragraph
        ])
    
    ranking_table = Table(ranking_data, colWidths=[rank_col_width, id_col_width, reward_col_width, tools_col_width, answer_col_width])
    ranking_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (3, -1), 'CENTER'),  # Center align first four columns
        ('ALIGN', (4, 0), (4, -1), 'LEFT'),    # Left align answer column
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(ranking_table)
    story.append(Spacer(1, 20))
    
    # Include round robin tournament results if available
    if round_robin_results:
        story.append(Paragraph("Round Robin Tournament Results:", heading_style))
        
        # Tournament summary
        summary_data = [["Completion ID", "Win Rate", "Wins", "Losses", "Ties", "Tool Calls"]]
        for comp_id, results in round_robin_results['summary'].items():
            # Find tool calls for this completion
            tc = 0
            for item in extracted_data:
                if item['id'] == int(comp_id):
                    tc = item['num_tool_calls']
                    break
                    
            summary_data.append([
                f"#{comp_id}",
                f"{results['win_rate']:.2f}%",
                str(results['wins']),
                str(results['losses']),
                str(results['ties']),
                str(tc)
            ])
        
        # Calculate column widths for summary table (now 6 columns)
        summary_col_width = int(available_width / 6)  # Equal widths for 6 columns
        
        summary_table = Table(summary_data, colWidths=[summary_col_width] * 6)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Individual matchups
        story.append(Paragraph("Individual Matchups:", subheading_style))
        
        # Calculate column widths for matchup tables
        principle_col_width = int(available_width * 0.25)    # 25%
        weight_col_width = int(available_width * 0.10)       # 10%
        winner_col_width = int(available_width * 0.15)       # 15%
        explanation_col_width = int(available_width * 0.50)  # 50%
        
        for match in round_robin_results['matchups']:
            # Get tool calls for both completions
            comp1_tools = "?"
            comp2_tools = "?"
            for item in extracted_data:
                if item['id'] == match['comp1']:
                    comp1_tools = str(item['num_tool_calls'])
                if item['id'] == match['comp2']:
                    comp2_tools = str(item['num_tool_calls'])
            
            story.append(Paragraph(
                f"Completion #{match['comp1']} ({comp1_tools} tools) vs Completion #{match['comp2']} ({comp2_tools} tools)", 
                section_style
            ))
            
            principles_data = [["Principle", "Weight", "Winner", "Explanation"]]
            for p in match['principles']:
                winner_text = f"#{match['comp1']}" if p['winner'] == 1 else (
                    f"#{match['comp2']}" if p['winner'] == 2 else "Tie"
                )
                
                # Create paragraph objects for wrapping text
                principle_para = Paragraph(p['principle'], table_cell_style)
                explanation_text = p.get('explanation', 'No explanation')
                explanation_para = Paragraph(escape_tags(explanation_text), table_cell_style)
                
                principles_data.append([
                    principle_para,
                    f"{p['weight']:.2f}",
                    winner_text,
                    explanation_para
                ])
            
            match_table = Table(principles_data, colWidths=[
                principle_col_width, weight_col_width, winner_col_width, explanation_col_width
            ])
            match_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),     # Left align principle column
                ('ALIGN', (1, 0), (2, -1), 'CENTER'),   # Center align weight and winner columns
                ('ALIGN', (3, 0), (3, -1), 'LEFT'),     # Left align explanation column
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            
            story.append(match_table)
            story.append(Paragraph(
                f"Overall Winner: {f'Completion #{match['comp1']}' if match['overall_winner'] == 1 else (f'Completion #{match['comp2']}' if match['overall_winner'] == 2 else 'Tie')}",
                normal_style
            ))
            story.append(Spacer(1, 15))
    
    doc.build(story)
