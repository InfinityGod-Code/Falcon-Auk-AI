from backend.tool_runtime_context import ToolRegistry
from backend.tools.open_ai.falcon_auk_tool import FalconAukTool


class ToolRegistryExecutor:
    def __init__(self, context: ToolRegistry):
        self.context = context

    def tool(self, name: str, description: str):
        def decorator(func):
            tool = FalconAukTool(
                name=name,
                description=description,
                func=func,
            )

            self.context.register_tool(tool)
            return tool

        return decorator

