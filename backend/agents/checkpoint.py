"""
Checkpoint system — save, restore, and manage agent execution snapshots.

Provides:
  - Checkpoint dataclass for serialisable state.
  - CheckpointStore (ABC) for pluggable storage backends.
  - InMemoryCheckpointStore for testing.
  - LocalFileCheckpointStore for JSON file persistence.
  - CheckpointManager for orchestration.
"""

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path


@dataclass
class Checkpoint:
    """
    A serializable snapshot of an agent's full state at a point in time.

    Does NOT store API keys — provider info is limited to type + model name.
    """

    checkpoint_id: str
    agent_name: str
    created_at: float = field(default_factory=time.time)

    # Provider info (no API key)
    provider_type: str = ""
    provider_model: str = ""

    # Serialized context
    context_data: dict[str, Any] = field(default_factory=dict)
    usage_data: dict[str, Any] = field(default_factory=dict)

    # User-defined metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Execution tracking
    run_count: int = 0

    def to_dict(self) -> dict:
        return {
            "checkpoint_id": self.checkpoint_id,
            "agent_name": self.agent_name,
            "created_at": self.created_at,
            "provider_type": self.provider_type,
            "provider_model": self.provider_model,
            "context_data": self.context_data,
            "usage_data": self.usage_data,
            "metadata": self.metadata,
            "run_count": self.run_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        return cls(
            checkpoint_id=data["checkpoint_id"],
            agent_name=data.get("agent_name", ""),
            created_at=data.get("created_at", time.time()),
            provider_type=data.get("provider_type", ""),
            provider_model=data.get("provider_model", ""),
            context_data=data.get("context_data", {}),
            usage_data=data.get("usage_data", {}),
            metadata=data.get("metadata", {}),
            run_count=data.get("run_count", 0),
        )


@dataclass
class CheckpointSummary:
    """Lightweight metadata for listing checkpoints without loading full state."""

    checkpoint_id: str
    agent_name: str
    created_at: float
    message_count: int
    run_count: int


class CheckpointStore(ABC):
    """Pluggable storage backend for checkpoints."""

    @abstractmethod
    def save(self, checkpoint_id: str, data: dict) -> None: ...

    @abstractmethod
    def load(self, checkpoint_id: str) -> dict: ...

    @abstractmethod
    def delete(self, checkpoint_id: str) -> None: ...

    @abstractmethod
    def list(self) -> list[CheckpointSummary]: ...


class InMemoryCheckpointStore(CheckpointStore):
    """In-memory store — useful for testing."""

    def __init__(self):
        self._store: dict[str, dict] = {}

    def save(self, checkpoint_id: str, data: dict):
        self._store[checkpoint_id] = data

    def load(self, checkpoint_id: str) -> dict:
        if checkpoint_id not in self._store:
            raise FileNotFoundError(f"Checkpoint '{checkpoint_id}' not found")
        return dict(self._store[checkpoint_id])

    def delete(self, checkpoint_id: str):
        self._store.pop(checkpoint_id, None)

    def list(self) -> list[CheckpointSummary]:
        summaries = []
        for cid, data in self._store.items():
            ctx = data.get("context_data", {})
            summaries.append(
                CheckpointSummary(
                    checkpoint_id=cid,
                    agent_name=data.get("agent_name", ""),
                    created_at=data.get("created_at", 0),
                    message_count=len(ctx.get("messages", [])),
                    run_count=data.get("run_count", 0),
                )
            )
        return summaries


class LocalFileCheckpointStore(CheckpointStore):
    """Persists checkpoints as JSON files in a directory."""

    def __init__(self, directory: str = "./checkpoints"):
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def _path(self, checkpoint_id: str) -> Path:
        safe = checkpoint_id.replace("/", "_").replace("\\", "_")
        return self._directory / f"{safe}.json"

    def save(self, checkpoint_id: str, data: dict):
        path = self._path(checkpoint_id)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load(self, checkpoint_id: str) -> dict:
        path = self._path(checkpoint_id)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        with open(path) as f:
            return json.load(f)

    def delete(self, checkpoint_id: str):
        path = self._path(checkpoint_id)
        if path.exists():
            path.unlink()

    def list(self) -> list[CheckpointSummary]:
        summaries = []
        for path in sorted(self._directory.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                ctx = data.get("context_data", {})
                summaries.append(
                    CheckpointSummary(
                        checkpoint_id=path.stem,
                        agent_name=data.get("agent_name", ""),
                        created_at=data.get("created_at", 0),
                        message_count=len(ctx.get("messages", [])),
                        run_count=data.get("run_count", 0),
                    )
                )
            except (json.JSONDecodeError, KeyError):
                continue
        return summaries


class CheckpointManager:
    """
    Orchestrates checkpoint creation, storage, and restoration.

    Can be attached to a BaseAgent for automatic checkpointing
    at key lifecycle points.
    """

    def __init__(self, store: CheckpointStore):
        self._store = store
        self._run_counter: int = 0

    # ── Core operations ─────────────────────────────────────────────

    def save(
        self,
        agent_name: str,
        context_data: dict,
        usage_data: dict,
        provider_type: str = "",
        provider_model: str = "",
        metadata: Optional[dict] = None,
        checkpoint_id: Optional[str] = None,
    ) -> Checkpoint:
        self._run_counter += 1

        cid = checkpoint_id or (
            f"ckpt_{agent_name}_{int(time.time())}_{self._run_counter}"
        )

        checkpoint = Checkpoint(
            checkpoint_id=cid,
            agent_name=agent_name,
            provider_type=provider_type,
            provider_model=provider_model,
            context_data=context_data,
            usage_data=usage_data,
            metadata=metadata or {},
            run_count=self._run_counter,
        )

        self._store.save(cid, checkpoint.to_dict())
        return checkpoint

    def load(self, checkpoint_id: str) -> Checkpoint:
        data = self._store.load(checkpoint_id)
        return Checkpoint.from_dict(data)

    def restore(
        self,
        checkpoint_id: str,
        context_manager: Any,
        usage_accumulator: Any,
    ) -> Checkpoint:
        """
        Restore context and usage from a checkpoint into live objects.
        """
        checkpoint = self.load(checkpoint_id)

        # Restore context
        if hasattr(context_manager, "replace_messages") and checkpoint.context_data:
            from backend.llm_providers.lifecycle import _message_from_dict

            messages = [
                _message_from_dict(m)
                for m in checkpoint.context_data.get("messages", [])
            ]
            context_manager.replace_messages(messages)

        # Restore usage
        if checkpoint.usage_data and hasattr(usage_accumulator, "add"):
            from backend.messages.usage import Usage

            usage = Usage.from_dict(checkpoint.usage_data)
            usage_accumulator.add(usage)

        return checkpoint

    def delete(self, checkpoint_id: str):
        self._store.delete(checkpoint_id)

    def list(self) -> list[CheckpointSummary]:
        return self._store.list()

    # ── Hook methods for automatic checkpointing ────────────────────

    def on_generation_end(self, **kwargs):
        """
        Intended to be wired into a CallbackManager so checkpoints
        are created automatically after every generation.
        """
        agent = kwargs.get("agent")
        if agent is None:
            return
        context_data = agent.context.to_dict() if hasattr(agent, "context") else {}
        usage_data = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if hasattr(agent, "total_usage"):
            u = agent.total_usage
            usage_data = {
                "prompt_tokens": u.prompt_tokens,
                "completion_tokens": u.completion_tokens,
                "total_tokens": u.total_tokens,
            }
        self.save(
            agent_name=agent.name,
            context_data=context_data,
            usage_data=usage_data,
            provider_type=agent.provider.provider.value
            if hasattr(agent, "provider")
            else "",
            provider_model=agent.provider.model if hasattr(agent, "provider") else "",
            metadata={"event": "generation_end"},
        )
