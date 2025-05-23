"""
Core concept: ArenaStage defines the domain and evaluation criteria for training your pico Deep Researcher.

How it works:
1. You define:
   - A list of questions that represent the types of reports you want to generate
   - A list of principles (with weights) that define what makes a good report

2. The system then:
   - Takes your questions and splits them into training/test sets
   - Has the LLM generate competing reports for each question
   - Uses another LLM as a judge to evaluate reports based on your principles
   - Uses the judge's decisions as training signals

Example: If you want reports that are both informative and entertaining, you would:
- Define questions like "Should schools teach coding?"
- Set principles like "Clear explanation of benefits" (weight: 1) and "Engaging writing style" (weight: 0.5)
- The system will train the LLM to balance these principles in its reports

See DebateArenaStage for a concrete implementation.
"""

import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Tuple, Any, List
from dataclasses import dataclass


class ArenaStage(ABC):
    """
    Base class for defining evaluation domains and criteria.

    To create a new evaluation domain, you only need to define:
    1. questions: List of questions that represent the types of reports you want to generate
       - Keep questions focused on the topic
       - Don't include style/format instructions in questions
       - Example: "Should schools teach coding?" (good)
       - Example: "Write a funny report about coding in schools" (bad - style in question)

    2. principles_and_weights: List of (principle, weight) tuples that define evaluation criteria
       - Each principle should be a clear, measurable quality
       - Weights determine how important each principle is
       - Example: ("Clear explanation of technical concepts", 1.0)
       - Example: ("Engaging writing style", 0.5)

    The system will:
    1. Split your questions into training (85%) and test (15%) sets
    2. For each question:
       - Generate two competing reports
       - Have an LLM judge evaluate them based on your principles
       - Use the judge's decision as a training signal
    """
    def __init__(self):
        # Override these in your implementation
        self.questions = [None]  # List of questions to generate reports for
        self.principles_and_weights = [(None,None)]  # List of (principle, weight) tuples



@dataclass
class DebateArenaStage(ArenaStage):
    """
    Example implementation: Debate-style reports that present arguments for/against topics.

    This stage:
    - Uses questions about policy/social topics that can be debated
    - Evaluates reports based on logical argument structure and reasoning
    - Could be extended with additional principles like:
      * Use of evidence and citations
      * Persuasiveness of arguments
      * Balance of perspectives
    """
    def __init__(self): 
        # Questions that can be debated from multiple perspectives
        self.questions = [
            "Video games should be taught as a school sport",
            "All schools should have mandatory cooking classes",
            "Homework should be replaced with project-based learning",
            "Every city should have a night market",
            "Movie theaters should have special quiet showings",
            "All schools should teach sign language",
            "Restaurants should offer smaller portion options",
            "Public spaces should have musical instruments",
            "All high schools should start after 9am",
            "Zoos should focus only on local wildlife",
            "Libraries should have recording studios",
            "Every workplace should allow pets",
            "Schools should teach financial literacy",
            "All restaurants should show calorie counts",
            "Museums should be open late on weekends",
            "Cities should have designated graffiti walls",
            "Schools should teach basic coding",
            "Grocery stores should have recipe stations",
            "All buildings should have rooftop gardens",
            "Cafes should have board game nights",
            "Libraries should offer virtual reality rooms",
            "Parks should have outdoor movie screens",
            "Schools should teach meditation",
            "Restaurants should compost food waste",
            "Cities should have more water fountains",
            "All schools should have maker spaces",
            "Gyms should offer childcare",
            "Libraries should loan art pieces",
            "Hotels should adopt shelter pets",
            "Schools should teach gardening",
            "Airports should have sleeping pods",
            "Malls should have indoor gardens",
            "Restaurants should grow their own herbs",
            "Cities should have free music venues",
            "Schools should teach public speaking",
            "Offices should have nap rooms",
            "Supermarkets should have tasting stations",
            "Libraries should have podcast studios",
            "Parks should have outdoor chess tables",
            "Schools should teach time management",
            "Restaurants should offer cooking classes",
            "Cities should have stargazing areas",
            "Beaches should have free sunscreen",
            "Schools should teach digital citizenship",
            "Hotels should have community spaces",
            "Parks should have fruit trees",
            "Libraries should offer language exchanges",
            "Theaters should have subtitle options",
            "Schools should teach environmental science",
            "Cities should have interactive art installations"
        ]
        
        # Currently only evaluating logical argument structure
        # Could be expanded to include more principles
        self.principles_and_weights = [
            ("The report presents a clear, well-structured logical argument with sound reasoning", 1),
        ]
        # Example of how to add more principles:
        # self.principles_and_weights = [
        #     ("The report presents a clear, well-structured logical argument with sound reasoning", 1),
        #     ("The report includes relevant, credible citations and evidence to support its claims", 1),
        #     ("The report is persuasive and effectively communicates its position", 1)
        # ]


def get_arena_stage(arenastage_name: str) -> ArenaStage:
    """
    Factory function to create ArenaStage instances.
    
    Args:
        arenastage_name: Name of the stage to create (currently only 'debate' supported)
    
    Returns:
        An instance of the requested ArenaStage
        
    Raises:
        ValueError: If the requested stage isn't supported
    """
    if arenastage_name.lower() == 'debate':
        return DebateArenaStage()
    else:
        raise ValueError(f"ArenaStage {arenastage_name} not supported. Currently 'debate' is available.")

