"""
Test MCP server — run as a stdio subprocess.
Provides a handful of demo tools for integration testing.

Usage:
    uv run python backend/mcp_integration/test_server.py
"""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("TestServer")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@mcp.tool()
def get_string_length(text: str) -> int:
    """Get the length of a string."""
    return len(text)


@mcp.tool()
def reverse_string(text: str) -> str:
    """Reverse a string."""
    return text[::-1]


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    conditions = {
        "paris": "21°C, rainy and cold",
        "london": "15°C, cloudy",
        "tokyo": "28°C, sunny",
        "new york": "18°C, partly cloudy",
        "sydney": "32°C, clear skies",
    }
    result = conditions.get(city.lower(), f"25°C, sunny")
    return f"The current weather in {city} is {result}."


if __name__ == "__main__":
    mcp.run(transport="stdio")
