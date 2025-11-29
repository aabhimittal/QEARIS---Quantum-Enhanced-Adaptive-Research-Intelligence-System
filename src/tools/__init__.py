"""
# ============================================================================
# QEARIS Tools Module
# ============================================================================
# 
# CAPSTONE REQUIREMENT: Tools Integration
# POINTS: Technical Implementation - 20 points
# 
# This module provides all tool-related components:
# 
# COMPONENTS:
# - MCP Server: Model Context Protocol implementation
# - Custom Tools: Research-specific tools (web search, arxiv, patents)
# - Built-in Tools: Google Search, code execution
# - OpenAPI Tools: Dynamic tool generation from OpenAPI specs
# 
# ARCHITECTURE:
# Tools are registered with the MCP server and can be invoked by agents
# through a standardized interface. This enables consistent tool usage
# across all agent types.
# ============================================================================
"""

# MCP Server is in core module (legacy location)
# New tools are provided here

from src.tools.builtin_tools import (BuiltinToolRegistry, calculator_tool,
                                     code_execution_tool, google_search_tool,
                                     register_builtin_tools)
from src.tools.custom_tools import (CustomToolRegistry, arxiv_search_tool,
                                    knowledge_base_tool, patent_search_tool,
                                    register_custom_tools, web_search_tool)
from src.tools.openapi_tools import (OpenAPITool, OpenAPIToolGenerator,
                                     generate_tools_from_spec)

__all__ = [
    # Custom tools
    "CustomToolRegistry",
    "web_search_tool",
    "arxiv_search_tool",
    "patent_search_tool",
    "knowledge_base_tool",
    "register_custom_tools",
    # Built-in tools
    "BuiltinToolRegistry",
    "google_search_tool",
    "code_execution_tool",
    "calculator_tool",
    "register_builtin_tools",
    # OpenAPI tools
    "OpenAPIToolGenerator",
    "generate_tools_from_spec",
    "OpenAPITool",
]
