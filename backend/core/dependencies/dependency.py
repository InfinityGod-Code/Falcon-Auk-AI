from dependency_injector import containers, providers
from backend.llm_providers.callback import CallbackManager
from backend.messages.usage import UsageAccumulator
from backend.agents.memory_context import MemoryContextManager
from backend.agents.checkpoint import CheckpointManager, InMemoryCheckpointStore
from backend.llm_providers.lifecycle import LLMLifecycle
from backend.llm_providers.openai import OpenAILLMProvider
from backend.tool_runtime_context import ToolRegistry


class Container(containers.DeclarativeContainer):
    tool_registry = providers.Factory(ToolRegistry)
    callback_manager = providers.Factory(CallbackManager)
    usage_accumulator = providers.Factory(UsageAccumulator)
    memory_context_manager = providers.Factory(MemoryContextManager)
    checkpoint_store = providers.Singleton(InMemoryCheckpointStore)
    checkpoint_manager = providers.Factory(
        CheckpointManager,
        store=checkpoint_store,
    )
    openai_provider = providers.Factory(OpenAILLMProvider)
    llm_lifecycle = providers.Factory(
        LLMLifecycle,
        usage_accumulator=usage_accumulator,
    )

container = Container()
