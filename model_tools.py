"""
Core module for model-tool integration.

Handles:
- Tool call detection and parsing
- Tool execution and response management
- Text processing with tool responses
- Response masking for training
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from tools import ToolRegistry, ToolResponse, Tool

logger = logging.getLogger(__name__)

class ToolCall:
    """
    Represents a single tool invocation in model output.
    
    Attributes:
        tool_name: Name of the tool to call
        parameters: Dictionary of tool parameters
    """
    def __init__(self, tool_name: str, parameters: Dict[str, Any]):
        self.tool_name = tool_name
        self.parameters = parameters

class ModelTools:
    """
    Manages interaction between language models and tools.
    
    Responsibilities:
    - Tool availability communication
    - Tool call detection and execution
    - Response processing and masking
    """
    
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry
    
    def get_tools_prompt(self) -> str:
        """
        Generate prompt section describing available tools.
        
        Returns:
            Formatted string containing tool descriptions and usage instructions
        """
        return f"""
You have access to the following tools:

{self.tool_registry.get_prompt_instructions()}

When you want to use a tool, format your request in XML format as shown above.
The tool's response will be provided in a <tool_response> tag.
"""
    
    def detect_tool_calls(self, text: str) -> List[Tuple[int, int, ToolCall]]:
        """
        Find and parse tool calls in model output.
        
        Args:
            text: Raw model output text
            
        Returns:
            List of (start_pos, end_pos, ToolCall) tuples for each valid tool call
        """
        # Match XML-formatted tool calls
        xml_pattern = r"<tool\s+name=\"([^\"]+)\"\s+([^>]+)>"
        matches = []
        
        for match in re.finditer(xml_pattern, text):
            try:
                tool_name = match.group(1)
                # Extract parameters from XML attributes
                params_str = match.group(2)
                params = {}
                for param in re.finditer(r'(\w+)="([^"]+)"', params_str):
                    params[param.group(1)] = param.group(2)
                
                tool_call = ToolCall(tool_name, params)
                matches.append((match.start(), match.end(), tool_call))
            except Exception as e:
                logger.warning(f"Failed to parse tool call: {e}")
                continue
        
        return matches
    
    def execute_tool_call(self, tool_call: ToolCall) -> ToolResponse:
        """
        Execute a tool call and handle its response.
        
        Args:
            tool_call: ToolCall instance to execute
            
        Returns:
            ToolResponse containing execution results or error
        """
        tool = self.tool_registry.get_tool(tool_call.tool_name)
        if not tool:
            return ToolResponse(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_call.tool_name}"
            )
        
        return tool.execute(**tool_call.parameters)
    
    def process_model_output(self, text: str) -> Tuple[str, List[ToolResponse]]:
        """
        Process model output by executing tool calls and inserting responses.
        
        Args:
            text: Raw model output text
            
        Returns:
            Tuple of (processed_text, tool_responses)
            - processed_text: Original text with tool responses inserted
            - tool_responses: List of responses from executed tools
        """
        tool_calls = self.detect_tool_calls(text)
        if not tool_calls:
            return text, []
        
        # Process tool calls in reverse order to maintain positions
        tool_calls.sort(key=lambda x: x[1], reverse=True)
        
        processed_text = text
        tool_responses = []
        
        for start, end, tool_call in tool_calls:
            # Execute tool and collect response
            response = self.execute_tool_call(tool_call)
            tool_responses.append(response)
            
            # Insert response after tool call
            processed_text = processed_text[:end] + response.to_xml() + processed_text[end:]
        
        return processed_text, tool_responses
    
    def mask_tool_responses(self, text: str) -> str:
        """
        Replace tool responses with mask tokens for training.
        
        Args:
            text: Text containing tool responses
            
        Returns:
            Text with tool responses replaced by mask tokens
        """
        # Replace all tool responses with mask token
        masked_text = re.sub(
            r"<tool_response[^>]*>.*?</tool_response>",
            "<TOOL_RESPONSE_MASK>",
            text,
            flags=re.DOTALL
        )
        return masked_text 