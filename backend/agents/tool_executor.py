import json

from backend.messages.tool_call import ToolCall
from backend.tool_registry import ToolRegistry

"""This is responsible for extracting the tool or function from the LLM Tool Response, parse and loads the 
   tools arguments to the function and return the result in the string format."""
class ToolExecutor:
    "ToolRegistry contains map where tool name is the key and instance is the FalconAukTool"
    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    
    def execute(self, tool_call: ToolCall) -> str:
        # name of the tool from the LLM tool call
        name = tool_call.function["name"]
        args = json.loads(tool_call.function["arguments"])

        tool = self._registry.get_tool(name)
        if tool is None:
            return f"Error: tool '{name}' not found"

        try:
            result = tool.func(**args)
            return str(result)
        except Exception as e:
            return f"Error executing '{name}': {e}"
