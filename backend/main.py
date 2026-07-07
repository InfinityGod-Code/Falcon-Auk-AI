import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# uv run python backend/main.py

from backend.core.decorators.tool_registry import ToolRegistryExecutor
from backend.core.dependencies.dependency import container
from backend.core.base.models.model import ModelProvider

tool_registration1 = container.tool_registry()
registry = ToolRegistryExecutor(tool_registration1)

tool_registration2 = container.tool_registry()
registry2 = ToolRegistryExecutor(tool_registration2)


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


@registry2.tool(
    name="third_sample_function",
    description="A third sample function that takes a string and an integer and returns a formatted string.",
)
def third_sample_function(param1: str, param2: int) -> str:
    """
    A third sample function that takes a string and an integer and returns a formatted string.
    """
    return f"Third function received string: {param1} and integer: {param2}"




print(
    f"✅ System Context: Current model provider is set to {tool_registration1.get_all_schemas(ModelProvider.OPENAI)}"
)
print(
    f"✅ System Context: Registered tools: {tool_registration2.get_all_schemas(ModelProvider.OPENAI)}"
)
