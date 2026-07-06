import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.dependencies.dependency import container
from backend.core.decorators.tool_registry import registry
from backend.core.base.models.model import ModelProvider


@registry.tool(
    name="sample_function",
    description="A sample function that takes a string and an integer and returns a formatted string.",
)
def sample_function(param1: str, param2: int) -> str:
    """
    A sample function that takes a string and an integer and returns a formatted string.
    """
    return f"Received string: {param1} and integer: {param2}"


ctx = container.context()

print(ctx.get_all_schemas(ModelProvider.OPENAI))
