"""
MCP Server - Model Context Protocol implementation

PURPOSE: Provide tools for AI agents
DESIGN: Tool registry with async execution
PROTOCOL: MCP-like interface for AI tool use
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """
    Tool definition for MCP

    DESIGN:
    - Name and description for LLM
    - Handler for execution
    - Parameters schema (optional)
    """

    name: str
    description: str
    handler: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class MCPRequest:
    """
    Request to execute a tool

    DESIGN:
    - Tool name
    - Parameters
    - Request metadata
    """

    tool_name: str
    parameters: Dict[str, Any]
    request_id: str = field(default_factory=lambda: str(datetime.now().timestamp()))
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MCPResponse:
    """
    Response from tool execution

    DESIGN:
    - Success flag
    - Result data
    - Error information
    """

    success: bool
    result: Any
    error: Optional[str] = None
    execution_time: float = 0.0
    request_id: str = ""


class MCPServer:
    """
    Model Context Protocol Server

    ARCHITECTURE:
    -------------
    1. Tool Registry: Stores available tools
    2. Executor: Runs tool handlers
    3. Error Handler: Manages failures

    WHY MCP?
    --------
    - Standard interface for AI tools
    - Separation of concerns
    - Easy tool addition
    - Consistent error handling

    USAGE:
    ------
    1. Register tools with register_tool()
    2. Execute with execute_tool()
    3. List with list_tools()
    """

    def __init__(self, config: Any = None):
        """
        Initialize MCP server

        PARAMETERS:
        -----------
        config: Application settings
        """
        if config:
            self.timeout = getattr(config, "MCP_TOOL_TIMEOUT", 60)
            self.retry_attempts = getattr(config, "MCP_RETRY_ATTEMPTS", 3)
        else:
            self.timeout = 60
            self.retry_attempts = 3

        # Tool registry
        self.tools: Dict[str, MCPTool] = {}

        # Execution metrics
        self.metrics = {"total_requests": 0, "successful": 0, "failed": 0, "total_time": 0.0}

        logger.info(
            f"MCPServer initialized: " f"timeout={self.timeout}s, " f"retries={self.retry_attempts}"
        )

    async def register_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Register a new tool

        PARAMETERS:
        -----------
        name: Unique tool name
        description: Human-readable description
        handler: Async function to execute
        parameters: Parameter schema

        RETURNS: True if registered
        """
        if name in self.tools:
            logger.warning(f"Tool '{name}' already registered, updating")

        tool = MCPTool(
            name=name, description=description, handler=handler, parameters=parameters or {}
        )

        self.tools[name] = tool

        logger.info(f"Registered tool: {name}")
        return True

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> MCPResponse:
        """
        Execute a registered tool

        PROCESS:
        1. Find tool
        2. Execute handler with timeout
        3. Handle errors
        4. Return response

        RETURNS: MCPResponse
        """
        start_time = datetime.now()
        self.metrics["total_requests"] += 1

        # Check tool exists
        if tool_name not in self.tools:
            logger.error(f"Tool not found: {tool_name}")
            self.metrics["failed"] += 1
            return MCPResponse(success=False, result=None, error=f"Tool '{tool_name}' not found")

        tool = self.tools[tool_name]

        if not tool.enabled:
            logger.warning(f"Tool disabled: {tool_name}")
            self.metrics["failed"] += 1
            return MCPResponse(success=False, result=None, error=f"Tool '{tool_name}' is disabled")

        # Execute with retry
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(tool.handler(**parameters), timeout=self.timeout)

                execution_time = (datetime.now() - start_time).total_seconds()
                self.metrics["successful"] += 1
                self.metrics["total_time"] += execution_time

                logger.debug(f"Tool executed: {tool_name} " f"({execution_time:.2f}s)")

                return MCPResponse(success=True, result=result, execution_time=execution_time)

            except asyncio.TimeoutError:
                last_error = f"Timeout after {self.timeout}s"
                logger.warning(
                    f"Tool timeout: {tool_name} " f"(attempt {attempt + 1}/{self.retry_attempts})"
                )

            except Exception as e:
                last_error = str(e)
                logger.error(
                    f"Tool error: {tool_name} - {str(e)} "
                    f"(attempt {attempt + 1}/{self.retry_attempts})"
                )

        # All retries failed
        self.metrics["failed"] += 1
        execution_time = (datetime.now() - start_time).total_seconds()

        return MCPResponse(
            success=False, result=None, error=last_error, execution_time=execution_time
        )

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all registered tools

        RETURNS: List of tool descriptions
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "enabled": tool.enabled,
            }
            for tool in self.tools.values()
        ]

    def enable_tool(self, name: str) -> bool:
        """Enable a tool"""
        if name in self.tools:
            self.tools[name].enabled = True
            logger.info(f"Tool enabled: {name}")
            return True
        return False

    def disable_tool(self, name: str) -> bool:
        """Disable a tool"""
        if name in self.tools:
            self.tools[name].enabled = False
            logger.info(f"Tool disabled: {name}")
            return True
        return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get execution metrics"""
        total = self.metrics["total_requests"]
        return {
            **self.metrics,
            "success_rate": self.metrics["successful"] / total if total > 0 else 0,
            "avg_time": self.metrics["total_time"] / total if total > 0 else 0,
            "registered_tools": len(self.tools),
            "enabled_tools": sum(1 for t in self.tools.values() if t.enabled),
        }

    async def close(self):
        """Cleanup resources"""
        logger.info("MCPServer shutting down")
        self.tools.clear()
