"""
Core module for tool management and execution.

Provides:
- Base tool interface and response structure
- Tool registry for centralized management
- Web search implementation using Brave API
- XML-based tool call parsing and response formatting
"""

import os
import yaml
import json
import logging
import requests
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ToolResponse:
    """
    Structured response from tool execution.
    
    Attributes:
        success: Whether the tool execution succeeded
        data: Response data (if successful)
        error: Error message (if failed)
    """
    success: bool
    data: Any
    error: Optional[str] = None

    def to_xml(self) -> str:
        """
        Convert response to XML format.
        
        Returns:
            XML string with either value or error attribute
        """
        if not self.success:
            return f'<tool_response error="{self.error}" />'
        
        # Convert data to string representation
        data_str = json.dumps(self.data) if isinstance(self.data, (dict, list)) else str(self.data)
        return f'<tool_response value="{data_str}" />'

class Tool(ABC):
    """
    Base interface for all tools.
    
    Required implementations:
    - name: Tool identifier
    - description: Tool functionality description
    - input_schema: Expected input format
    - execute: Core tool functionality
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool identifier used in XML calls."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of tool functionality."""
        pass
    
    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """JSON schema defining required input parameters."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResponse:
        """
        Execute tool with provided parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResponse with execution results
        """
        pass
    
    def get_prompt_instruction(self) -> str:
        """
        Generate usage instructions for prompts.
        
        Returns:
            Formatted string with tool name, description, and usage example
        """
        return f"""
Tool: {self.name}
Description: {self.description}
Usage: Call the tool using XML format like this:
<tool name="{self.name}" {self._format_schema_for_xml()}></tool>
The tool will respond with its results in a <tool_response> tag.
"""

    def _format_schema_for_xml(self) -> str:
        """
        Convert input schema to XML attribute format.
        
        Returns:
            Space-separated string of attribute placeholders
        """
        attrs = []
        for key, value in self.input_schema.items():
            attrs.append(f'{key}="value"')
        return " ".join(attrs)

    @staticmethod
    def parse_xml_tool_call(xml_str: str) -> Tuple[str, Dict[str, str]]:
        """
        Parse XML tool call into components.
        
        Args:
            xml_str: XML-formatted tool call
            
        Returns:
            Tuple of (tool_name, parameters)
            
        Raises:
            ValueError: If XML is invalid or missing required attributes
        """
        try:
            root = ET.fromstring(xml_str)
            if root.tag != 'tool':
                raise ValueError("Invalid tool call format - must start with <tool>")
            
            tool_name = root.get('name')
            if not tool_name:
                raise ValueError("Tool name is required")
            
            # Extract all attributes as parameters
            params = {k: v for k, v in root.attrib.items() if k != 'name'}
            
            return tool_name, params
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {str(e)}")

class BraveSearchTool(Tool):
    """
    Web search implementation using Brave Search API.
    
    Features:
    - Configurable result count
    - Structured response format
    - Error handling
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
        
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web using Brave Search API. Returns up to 5 relevant news articles with their sources and descriptions."
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "query": "string"  # The search query
        }
    
    def get_prompt_instructions(self) -> str:
        return """
        You can search the web using the Brave Search API. To use this tool, call it with a query parameter:
        <tool name="web_search" query="your search query here"></tool>
        """
        
    def execute(self, **kwargs) -> ToolResponse:
        """
        Execute web search with provided query.
        
        Args:
            **kwargs: Must contain 'query' parameter
            
        Returns:
            ToolResponse with search results or error
        """
        if 'query' not in kwargs:
            return ToolResponse(success=False, error="Missing required parameter: query")
            
        query = kwargs['query']
        
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        
        params = {
            "q": query,
            "count": 2  # Fixed count of 5 results
        }
        
        try:
            response = requests.get(self.base_url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant information from the response
            results = []
            for result in data.get('web', {}).get('results', []):
                results.append({
                    'title': result.get('title', ''),
                    'description': result.get('description', ''),
                    'url': result.get('url', '')
                })
            
            return ToolResponse(success=True, data=results)
            
        except Exception as e:
            return ToolResponse(success=False, error=str(e))

class ToolRegistry:
    """
    Central registry for tool management.
    
    Features:
    - Tool registration and lookup
    - Combined prompt instructions
    - XML-based tool execution
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """
        Register a new tool.
        
        Args:
            tool: Tool instance to register
        """
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """
        Retrieve tool by name.
        
        Args:
            name: Tool identifier
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)
    
    def get_all_tools(self) -> List[Tool]:
        """
        Get all registered tools.
        
        Returns:
            List of all registered tool instances
        """
        return list(self._tools.values())
    
    def get_prompt_instructions(self) -> str:
        """
        Get combined instructions for all tools.
        
        Returns:
            Concatenated prompt instructions from all tools
        """
        return "\n".join(tool.get_prompt_instruction() for tool in self._tools.values())
    
    def execute_xml_tool_call(self, xml_str: str) -> str:
        """
        Execute tool call from XML string.
        
        Args:
            xml_str: XML-formatted tool call
            
        Returns:
            XML-formatted tool response
        """
        try:
            tool_name, params = Tool.parse_xml_tool_call(xml_str)
            tool = self.get_tool(tool_name)
            
            if not tool:
                return f'<tool_response error="Tool {tool_name} not found" />'
            
            response = tool.execute(**params)
            return response.to_xml()
            
        except ValueError as e:
            return f'<tool_response error="{str(e)}" />'
        except Exception as e:
            logger.error(f"Error executing tool call: {str(e)}")
            return f'<tool_response error="Internal error: {str(e)}" />'

def create_tool_registry(tool_config: str = "websearch") -> ToolRegistry:
    """
    Create tool registry with specified configuration.
    
    Args:
        tool_config: Configuration type (currently only "websearch" supported)
        
    Returns:
        Configured ToolRegistry instance
        
    Raises:
        ValueError: If configuration is not supported
    """
    registry = ToolRegistry()
    
    if tool_config == "websearch":
        registry.register(BraveSearchTool(os.getenv("BRAVE_API_KEY")))
    else:
        raise ValueError(f"Unknown tool configuration: {tool_config}")
    
    return registry 