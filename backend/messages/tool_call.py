class ToolCall:
    def __init__(self, id: str, name: str, arguments: str, type: str = "function"):
        self.id = id
        self.type = type
        self.function = {"name": name, "arguments": arguments}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function["name"],
                "arguments": self.function["arguments"],
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCall":
        return cls(
            id=data["id"],
            name=data["function"]["name"],
            arguments=data["function"]["arguments"],
            type=data.get("type", "function"),
        )
