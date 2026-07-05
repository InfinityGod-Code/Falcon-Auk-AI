import inspect
from backend.core.base.tools.tool import Tool


class OpenAITool(Tool) :

    def __init__(self,name,description,func) :
        super().__init__(name,description,func)

    def to_model_specific(self):

        """
        Dynamically extracts properties and type hints from the passed function
        and converts it into the exact JSON format OpenAI requires.
        """
        properties = {}
        required = []

        # Map Python native type annotations to standard JSON Schema types
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }

        # Loop through each parameter extracted by inspect.signature
        for param_name, param in self.signature.parameters.items():
            # Fallback to "string" if the developer didn't provide a type hint
            param_type = type_map.get(param.annotation, "string")

            properties[param_name] = {
                "type": param_type,
                "description": f"The {param_name} argument."  # Or pull from docstrings later
            }

            # If the parameter has no default value, it is a required field
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        # Construct the exact OpenAI Tool representation schema
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)



