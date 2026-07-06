import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

#uv run python backend/main.py

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

@registry.tool(
    name="another_sample_function",
    description="Another sample function that takes a string and an integer and returns a formatted string.",
)
def another_sample_function(param1: str, param2: int) -> str:
    """
    Another sample function that takes a string and an integer and returns a formatted string.
    """
    return f"Another function received string: {param1} and integer: {param2}"


@registry.tool(
    name="third_sample_function",
    description="A third sample function that takes a string and an integer and returns a formatted string.",
)
def third_sample_function(param1: str, param2: int) -> str:
    """
    A third sample function that takes a string and an integer and returns a formatted string.
    """
    return f"Third function received string: {param1} and integer: {param2}"


ctx = container.context()

print(ctx.get_all_schemas(ModelProvider.OPENAI))
