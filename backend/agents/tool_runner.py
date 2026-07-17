import json

from backend.messages.tool_call import ToolCall
from backend.tool_registry import ToolRegistry
from backend.mcp_integration.mcp_tool import MCPTool


class ToolRunner:
    """
    Responsible for resolving a tool call from the LLM into an actual
    function execution.  Supports both sync FalconAukTool instances
    and async MCPTool instances.
    """

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    async def execute(self, tool_call: ToolCall) -> str:
        name = tool_call.function["name"]
        args = json.loads(tool_call.function["arguments"])

        tool = self._registry.get_tool(name)
        if tool is None:
            return f"Error: tool '{name}' not found"

        try:
            if isinstance(tool, MCPTool):
                result = await tool.async_run(**args)
            else:
                result = tool.func(**args)
            return str(result)
        except Exception as e:
            return f"Error executing '{name}': {e}"
