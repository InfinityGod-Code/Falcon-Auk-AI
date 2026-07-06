import inspect
from backend.core.base.models.model import ModelProvider
from backend.core.base.tools.tool import Tool


class FalconAukTool(Tool):
    def __init__(self, name, description, func):
        super().__init__(name, description, func)

    def to_model_specific(self, model_provider: ModelProvider):

        match model_provider:
            case ModelProvider.OPENAI:
                return self._to_openai_tool()
            case ModelProvider.ANTHROPIC:
                raise NotImplementedError(
                    "Anthropic tool conversion is not implemented yet."
                )
            case ModelProvider.GEMINI:
                raise NotImplementedError(
                    "Gemini tool conversion is not implemented yet."
                )
            case _:
                raise ValueError(f"Unsupported model provider: {model_provider}")

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def _to_openai_tool(self):
        _type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }

        sig = inspect.signature(self.func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            raw_type = (
                param.annotation if param.annotation != inspect.Parameter.empty else str
            )
            json_type = _type_map.get(raw_type, "string")

            properties[name] = {
                "type": json_type,
            }

            if param.default is inspect.Parameter.empty:
                required.append(name)

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
