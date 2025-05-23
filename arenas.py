"""
Core environment that manages the entire training + evaluation setup.
Handles:
- Question prompts and templates
- Available tools for answering questions
- Training and test data splits
"""

import numpy as np
from tools import ToolRegistry
from arenastages import ArenaStage

class Arena(): 
    def __init__(self, stage: ArenaStage, tools: ToolRegistry): 
        # stage: Contains the questions and evaluation criteria
        # tools: Registry of available tools for answering questions
        self.stage = stage 
        self.tools = tools 
        self._setup_prompts()
        self._setup_train_test_split()

    def _setup_prompts(self): 
        # Template for generating answers to questions
        # Requires:
        # - {question}: The question to answer
        # - {tool_list_description}: List of available tools
        self.generation_prompt = """
        Please develop a 2-3 paragraph report to answer the following question: {question}. 

        You must answer by first reasoning about your answer in <think> </think> tags. 
        Then your final report must be contained within <answer></answer> tags. 

        Further, to aid you in your report you have access to the following tools {tool_list_description}. 

        A tool can only be called within the <think></think> portion of your response. And must be called in XML format.
        The tool response will be provided immediately after your tool call. 

        For example here is a short, example response: 

        <think> the user wants a report on X, let me first query the web <tool name="web_search" query="tell me more about x"></tool>
        <tool_response value="X is ......" />
        Oh interesting, let me analyze one more thing given this new information <tool name="web_search" query="tell me more about y"></tool>
        <tool_response value="y is ......" />
        I am now ready to write my report 
        </think> 

        <answer> 
        X is.... (2-3 paragraph nice report about/answering X) 
        </answer> 

        Now, again, please develop a 2-3 paragraph report to answer the following question: {question}. 
        YOUR FINAL ANSWER MUST BE CONTAINED WITHIN <answer></answer> TAGS.
        """

        # Template for comparing two generated answers
        # Requires:
        # - {question}: Original question
        # - {report_1}: First generated answer
        # - {report_2}: Second generated answer
        self.judging_prompt = """
        You will be presented with two reports generated to answer {question}. 

        Report 1: {report_1}

        Report 2: {report_2} 

        """

    def _setup_train_test_split(self):
        """Splits questions into training (85%) and test (15%) sets."""
        questions = self.stage.questions
        np.random.shuffle(questions)
        
        split_idx = int(len(questions) * 0.8)
        self.train_questions = questions[:split_idx]
        self.test_questions = questions[split_idx:]
        
        self.test_idx = 0  # Tracks position in test set

    def get_train_example(self) -> str:
        """Returns a random question from the training set."""
        return np.random.choice(self.train_questions)

    def get_test_example(self) -> str:
        """Returns the next question from the test set, cycling back to start when finished."""
        if self.test_idx >= len(self.test_questions):
            self.test_idx = 0
        example = self.test_questions[self.test_idx]
        self.test_idx += 1
        return example

    def reset_test_iterator(self):
        """Resets the test set position back to the first question."""
        self.test_idx = 0
