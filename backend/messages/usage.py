from dataclasses import dataclass, field


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict) -> "Usage":
        return cls(
            prompt_tokens=data.get("prompt_tokens", 0),
            completion_tokens=data.get("completion_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
        )

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


class UsageAccumulator:
    def __init__(self):
        self._usages: list[Usage] = []

    def add(self, usage: Usage):
        self._usages.append(usage)

    @property
    def total(self) -> Usage:
        if not self._usages:
            return Usage()
        return sum(self._usages[1:], self._usages[0])

    def reset(self):
        self._usages.clear()
