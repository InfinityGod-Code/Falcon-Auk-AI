from backend.application_context import ApplicationContext
from backend.core.dependencies.dependency import container
from backend.tools.open_ai.falcon_auk_tool import FalconAukTool


class ToolRegistry:
    def __init__(self, context: ApplicationContext):
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


registry = ToolRegistry(container.context())
