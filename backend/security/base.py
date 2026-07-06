from abc import ABC
from enum import Enum

class SecurityLevel(Enum):
    LOG_ONLY = "log_only"
    WARN = "warn"
    REDACT = "redact"
    REJECT = "reject"


class SecurityViolation(Exception):
    def __init__(self, message: str, handler: str, level: SecurityLevel):
        self.handler = handler
        self.level = level
        super().__init__(f"[{level.value.upper()}] {handler}: {message}")


class SecurityHandler(ABC):
    """
    Base class for all security handlers.

    Every handler has a level that controls behaviour on violation:
      LOG_ONLY — log the issue, continue execution
      WARN     — log + print warning, continue
      REDACT   — modify data in-place, continue
      REJECT   — raise SecurityViolation, halt execution

    Subclasses override the relevant on_* hooks from BaseCallbackHandler
    and call self.reject() / self.warn() as needed.
    """

    def __init__(self, level: SecurityLevel = SecurityLevel.REJECT):
        self.level = level
        self._violations: list[SecurityViolation] = []

    def reject(self, reason: str):
        violation = SecurityViolation(reason, type(self).__name__, self.level)
        self._violations.append(violation)
        if self.level == SecurityLevel.REJECT:
            raise violation

    def warn(self, reason: str):
        violation = SecurityViolation(reason, type(self).__name__, SecurityLevel.WARN)
        self._violations.append(violation)
        print(f"⚠️  {violation}")

    @property
    def violations(self) -> list[SecurityViolation]:
        return list(self._violations)

    def clear_violations(self):
        self._violations.clear()
