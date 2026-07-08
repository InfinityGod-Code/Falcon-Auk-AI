from typing import Dict, List, Optional
from backend.core.base.tools.tool import Tool
from backend.core.base.models.model import ModelProvider


class ToolRegistry:
    def __init__(self):
        self._registry: Dict[str, Tool] = {}
        self.current_model_provider: str = (
            None  # This will hold the current model provider
        )

    def register_tool(self, tool_instance: Tool):
        self._registry[tool_instance.name] = tool_instance

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._registry.get(name)

    def get_all_schemas(self, target_provider: ModelProvider) -> List[Dict]:
        return [t.to_model_specific(target_provider) for t in self._registry.values()]
