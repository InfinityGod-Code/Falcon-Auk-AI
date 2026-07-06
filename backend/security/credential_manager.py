import os
from pathlib import Path
from typing import Optional


class CredentialManager:
    """
    Manages API keys and secrets securely.

    Loading order (first found wins):
      1. Environment variable
      2. .env file
      3. Keychain / OS secret store (future)

    API keys are NEVER persisted in checkpoints or serialized state.
    """

    def __init__(self, env_prefix: str = "FALCON_AUK"):
        self._env_prefix = env_prefix
        self._dotenv_path = Path(".env")
        self._cache: dict[str, str] = {}
        self._loaded_env = False

    def _load_dotenv(self):
        if self._loaded_env or not self._dotenv_path.exists():
            return
        self._loaded_env = True
        with open(self._dotenv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("\"'")
                os.environ.setdefault(key, value)

    def get_api_key(self, provider: str = "OPENAI") -> Optional[str]:
        cache_key = f"{self._env_prefix}_{provider}_API_KEY"
        if cache_key in self._cache:
            return self._cache[cache_key]

        self._load_dotenv()

        key = os.environ.get(f"{self._env_prefix}_{provider}_API_KEY")
        if key:
            self._cache[cache_key] = key
            return key

        key = os.environ.get(f"{provider}_API_KEY")
        if key:
            self._cache[cache_key] = key
            return key

        return None

    def get_model(self, provider: str = "OPENAI") -> Optional[str]:
        return os.environ.get(
            f"{self._env_prefix}_{provider}_MODEL",
            os.environ.get(f"{provider}_MODEL"),
        )

    @staticmethod
    def sanitize(value: str, visible_chars: int = 4) -> str:
        if len(value) <= visible_chars:
            return "****"
        return value[:visible_chars] + "****" + value[-4:]
