"""
Core reward evaluation system for training Deep Research models.

The system evaluates model outputs using:
1. Principle-based scoring: Compares reports based on defined principles
2. Format validation: Ensures proper XML structure and content placement
3. Round-robin tournament: During training, compares all outputs against each other
4. Head-to-head comparison: During testing, compares against baseline model

Each evaluation produces:
- Numerical rewards for training
- Detailed metrics for analysis
- Principle-by-principle breakdowns
"""

import re
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any, Optional

from tqdm import tqdm

class RewardEvaluator(ABC):
    """
    Base interface for reward computation in RL training.
    
    Implementations must provide:
    1. compute_rewards: Score model outputs and return rewards + metrics
    2. get_reward_breakdown: Convert raw scores to labeled metrics
    """
    
    @abstractmethod
    def compute_rewards(
        self,
        prompts: List[List[Dict[str, str]]],
        completions: List[List[Dict[str, str]]],
        answer: Any,
        device: str
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Score a batch of model completions.
        
        Args:
            prompts: List of input prompts in chat format
            completions: List of model outputs in chat format
            answer: Ground truth for validation
            device: Compute device ("cpu" or "cuda")
            
        Returns:
            rewards_per_func: Tensor of shape (num_completions, num_reward_functions)
            metrics: Dictionary of aggregated metrics
        """
        pass

    @abstractmethod
    def get_reward_breakdown(self, reward_scores: torch.Tensor) -> Dict[str, float]:
        """
        Convert raw reward scores to labeled metrics.
        
        Args:
            reward_scores: Raw scores from compute_rewards
            
        Returns:
            Dictionary mapping metric names to values
        """
        pass




class GenericEvaluator(RewardEvaluator):
    """
    Evaluates reports using principles from ArenaStage.
    
    Scoring process:
    1. For each principle, judge model compares reports
    2. Winner gets principle's weight as reward
    3. Format rewards ensure proper XML structure
    4. Total reward = principle scores + format scores
    """
    def __init__(self, arena_stage):
        self.arena_stage = arena_stage
        self.principles_and_weights = arena_stage.principles_and_weights
        self.num_reward_functions = 4  # 1 principle + 3 format rewards

    def _extract_answer(self, text: str) -> str:
        """Extract content between <answer> tags."""
        try:
            answer = text.split("<answer>")[-1]
            answer = answer.split("</answer>")[0]
            return answer.strip()
        except Exception:
            return "Couldn't find answer - automatic failure"

    def _strict_format_reward(self, completions) -> List[float]:
        """Reward for exact XML format with newlines."""
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
        matches = [bool(re.match(pattern, r)) for r in completions]
        return [0.5 if m else 0.0 for m in matches]

    def _soft_format_reward(self, completions) -> List[float]:
        """Reward for basic XML structure."""
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        matches = [bool(re.match(pattern, r)) for r in completions]
        return [0.5 if m else 0.0 for m in matches]

    def _xml_count_reward(self, completions) -> List[float]:
        """Reward for proper XML tag usage."""
        def count_xml(text: str) -> float:
            count = 0.0
            if "<think>" in text: count += 0.125
            if "</think>" in text: count += 0.125
            if "<answer>" in text: count += 0.125
            if "</answer>" in text: count += 0.125
            # Penalize content after final tag
            if "</answer>" in text:
                count -= len(text.split("</answer>")[-1].strip())*0.001
            return count
        return [count_xml(r) for r in completions]

    def _judge_pair(self, judge_model, question, report1, report2, principle, system_prompt=None):
        """
        Have judge model compare two reports based on a principle.
        
        Returns:
            winner_code: 1 (report1), 2 (report2), or 0 (tie)
            explanation: Judge's reasoning
        """
        judge_prompt = f"""
You will be shown two reports answering the same question.

Question: {question}

Report 1:
{report1}

Report 2:
{report2}

Principle: {principle}

First, provide a brief, one-sentence explanation for your choice.
Then, on a new line, state which report better fulfills the above principle. Respond with EXACTLY one of these options:
- REPORT_1_WINS
- REPORT_2_WINS
"""
        system_prompt = system_prompt or "You are an impartial judge."
        judge_response_full = judge_model.generate(
            system_prompt=system_prompt,
            user_prompt=judge_prompt,
            max_new_tokens=150,
            temperature=0.1
        ).strip()

        explanation = "No explanation provided."
        winner_line = ""

        lines = judge_response_full.split('\\n')
        if len(lines) > 1:
            explanation = lines[0].strip()
            winner_line = lines[-1].strip().upper()
        elif len(lines) == 1:
            winner_line = lines[0].strip().upper()

        if "REPORT_1_WINS" in winner_line:
            return 1, explanation
        elif "REPORT_2_WINS" in winner_line:
            return 2, explanation
        else:
            return 0, judge_response_full

    def _compute_train_rewards(self, input_prompt, all_models, train_model_completions, device):
        """
        Compute rewards during training using round-robin tournament.
        
        Each completion is compared against all others for each principle.
        Winners get points based on principle weights.
        """
        num_completions = len(train_model_completions)
        rollout_scores = torch.zeros(num_completions, device=device)
        detailed_pairwise_judgements = []

        # Round-robin tournament
        for i in tqdm(range(num_completions), desc="Evaluating training completions", leave=False):
            for j in range(i + 1, num_completions):
                principle_evaluations_for_pair = []
                score_i_for_pair = 0
                score_j_for_pair = 0

                # Compare on each principle
                for principle, weight in self.principles_and_weights:
                    report1_text = self._extract_answer(train_model_completions[i])
                    report2_text = self._extract_answer(train_model_completions[j])
                    
                    winner_code, explanation = self._judge_pair(
                        all_models["judge_model"],
                        input_prompt,
                        report1_text,
                        report2_text,
                        principle
                    )
                    
                    # Update scores based on winner
                    if winner_code == 1:
                        score_i_for_pair += weight
                    elif winner_code == 2:
                        score_j_for_pair += weight

                    principle_evaluations_for_pair.append({
                        "principle": principle,
                        "weight": weight,
                        "winner_code": winner_code,
                        "winner_for_principle": f"rollout_{i}" if winner_code == 1 else 
                                              f"rollout_{j}" if winner_code == 2 else "tie",
                        "explanation": explanation
                    })

                # Record overall winner for this pair
                if score_i_for_pair > score_j_for_pair:
                    rollout_scores[i] += 1
                elif score_j_for_pair > score_i_for_pair:
                    rollout_scores[j] += 1
                else:
                    rollout_scores[i] += 0.5
                    rollout_scores[j] += 0.5

                detailed_pairwise_judgements.append({
                    "rollout_1_index": i,
                    "rollout_2_index": j,
                    "rollout_1_text_preview": train_model_completions[i][:100] + "...",
                    "rollout_2_text_preview": train_model_completions[j][:100] + "...",
                    "principle_evaluations": principle_evaluations_for_pair,
                    "overall_winner_for_pair": f"rollout_{i}" if score_i_for_pair > score_j_for_pair else
                                             f"rollout_{j}" if score_j_for_pair > score_i_for_pair else "tie",
                    "score_rollout_1_for_pair": score_i_for_pair,
                    "score_rollout_2_for_pair": score_j_for_pair
                })

        # Combine principle scores with format rewards
        rewards_per_func = torch.zeros(num_completions, self.num_reward_functions, device=device)
        rewards_per_func[:, 0] = rollout_scores
        rewards_per_func[:, 1] = torch.tensor(self._strict_format_reward(train_model_completions), device=device)
        rewards_per_func[:, 2] = torch.tensor(self._soft_format_reward(train_model_completions), device=device)
        rewards_per_func[:, 3] = torch.tensor(self._xml_count_reward(train_model_completions), device=device)

        # Compile metrics
        metrics = {
            "rewards/principles": rewards_per_func[:, 0].mean().item(),
            "rewards/strict_format": rewards_per_func[:, 1].mean().item(),
            "rewards/soft_format": rewards_per_func[:, 2].mean().item(),
            "rewards/xml_count": rewards_per_func[:, 3].mean().item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "detailed_pairwise_judgements": detailed_pairwise_judgements
        }
        return rewards_per_func, metrics

    def _compute_test_rewards(self, prompt, all_models, train_model_completions, compare_model_completions, device):
        """
        Compute rewards during testing using head-to-head comparison.
        
        Each training completion is compared against its corresponding baseline completion.
        Training model gets positive reward for wins, negative for losses.
        """
        num_comparisons = len(train_model_completions)
        rewards_per_func = torch.zeros(num_comparisons, self.num_reward_functions, device=device)
        all_principle_judgements = []

        # Head-to-head comparison
        for i in range(num_comparisons):
            current_comparison_judgements = []
            for principle, weight in self.principles_and_weights:
                report1 = self._extract_answer(train_model_completions[i])
                report2 = self._extract_answer(compare_model_completions[i])
                winner, explanation = self._judge_pair(
                    all_models["judge_model"],
                    prompt,
                    report1,
                    report2,
                    principle
                )
                
                current_comparison_judgements.append({
                    "principle": principle,
                    "weight": weight,
                    "winner_code": winner,
                    "explanation": explanation
                })

                # Update rewards based on winner
                if winner == 1:
                    rewards_per_func[i, 0] += weight
                elif winner == 2:
                    rewards_per_func[i, 0] -= weight

            all_principle_judgements.append(current_comparison_judgements)

        # Add format rewards
        rewards_per_func[:, 1] = torch.tensor(self._strict_format_reward(train_model_completions), device=device)
        rewards_per_func[:, 2] = torch.tensor(self._soft_format_reward(train_model_completions), device=device)
        rewards_per_func[:, 3] = torch.tensor(self._xml_count_reward(train_model_completions), device=device)

        # Compile metrics
        metrics = {
            "rewards/principles": rewards_per_func[:, 0].mean().item(),
            "rewards/strict_format": rewards_per_func[:, 1].mean().item(),
            "rewards/soft_format": rewards_per_func[:, 2].mean().item(),
            "rewards/xml_count": rewards_per_func[:, 3].mean().item(),
            "reward": rewards_per_func.sum(dim=1).mean().item(),
            "principle_judgements": all_principle_judgements
        }
        return rewards_per_func, metrics

    def get_round_robin_results(self):
        """
        Format round-robin tournament results for analysis.
        
        Returns:
            Dict containing:
            - summary: Win rates and match statistics per completion
            - matchups: Detailed results of each comparison
        """
        if not hasattr(self, '_last_detailed_judgements'):
            if hasattr(self, '_last_metrics') and 'detailed_pairwise_judgements' in self._last_metrics:
                self._last_detailed_judgements = self._last_metrics['detailed_pairwise_judgements']
            else:
                return None
        
        results = {
            'summary': {},
            'matchups': []
        }
        
        # Calculate completion statistics
        completion_stats = {}
        for match in self._last_detailed_judgements:
            comp1 = match['rollout_1_index'] + 1
            comp2 = match['rollout_2_index'] + 1
            
            for comp_id in [comp1, comp2]:
                if comp_id not in completion_stats:
                    completion_stats[comp_id] = {'wins': 0, 'losses': 0, 'ties': 0}
            
            # Record match outcome
            if match['overall_winner_for_pair'] == f"rollout_{match['rollout_1_index']}":
                completion_stats[comp1]['wins'] += 1
                completion_stats[comp2]['losses'] += 1
                overall_winner = 1
            elif match['overall_winner_for_pair'] == f"rollout_{match['rollout_2_index']}":
                completion_stats[comp1]['losses'] += 1
                completion_stats[comp2]['wins'] += 1
                overall_winner = 2
            else:
                completion_stats[comp1]['ties'] += 1
                completion_stats[comp2]['ties'] += 1
                overall_winner = 0
            
            # Format principle evaluations
            principles_data = [{
                'principle': eval_info['principle'],
                'weight': eval_info['weight'],
                'winner': 1 if eval_info['winner_code'] == 1 else (2 if eval_info['winner_code'] == 2 else 0),
                'explanation': eval_info['explanation']
            } for eval_info in match['principle_evaluations']]
            
            results['matchups'].append({
                'comp1': comp1,
                'comp2': comp2,
                'principles': principles_data,
                'overall_winner': overall_winner
            })
        
        # Calculate win rates
        for comp_id, stats in completion_stats.items():
            total_matches = stats['wins'] + stats['losses'] + stats['ties']
            win_rate = ((stats['wins'] + 0.5 * stats['ties']) / total_matches * 100) if total_matches > 0 else 0
            
            results['summary'][comp_id] = {
                'win_rate': win_rate,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'ties': stats['ties']
            }
        
        return results

    def compute_rewards(self, input_prompt, all_models, train_model_completions, compare_model_completions=None, device="cuda", is_test=False):
        """
        Main entry point for reward computation.
        
        Args:
            input_prompt: Question to evaluate
            all_models: Dict containing judge model
            train_model_completions: Outputs from model being trained
            compare_model_completions: Baseline outputs (required for testing)
            device: Compute device
            is_test: Whether this is a test evaluation
            
        Returns:
            rewards_per_func: Tensor of rewards
            metrics: Dictionary of evaluation metrics
        """
        if is_test:
            rewards, metrics = self._compute_test_rewards(input_prompt, all_models, train_model_completions, compare_model_completions, device)
        else:
            rewards, metrics = self._compute_train_rewards(input_prompt, all_models, train_model_completions, device)
        
        # Cache metrics for round-robin analysis
        self._last_metrics = metrics
        if 'detailed_pairwise_judgements' in metrics:
            self._last_detailed_judgements = metrics['detailed_pairwise_judgements']
        
        return rewards, metrics

    def get_reward_breakdown(self, rewards: torch.Tensor) -> Dict[str, float]:
        """
        Convert raw reward tensor to labeled metrics.
        
        Args:
            rewards: Raw reward tensor
            
        Returns:
            Dictionary mapping metric names to values
        """
        return {
            "principles": rewards[0].item(),
            "strict_format": rewards[1].item(),
            "soft_format": rewards[2].item(),
            "xml_count": rewards[3].item(),
            "reward": rewards.sum().item()
        }


