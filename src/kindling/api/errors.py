"""What a client is told when a model call fails, without leaking internals.

Raw errors from Ollama or an upstream server can name internal hosts, ports and model lists, so they
go to the log; clients get the error's kind. Errors written for clients pass through unchanged.
"""

from __future__ import annotations

import logging

log = logging.getLogger("kindling.errors")

# Exceptions whose messages are written for clients (by class name, so this module imports nothing).
_SAFE = {"StepLimitReached", "Overloaded", "ModelNotAllowed", "UnknownModel"}


def public_error(error: BaseException) -> str:
    if any(cls.__name__ in _SAFE for cls in type(error).__mro__):
        return str(error) or type(error).__name__
    log.warning("model call failed: %s: %s", type(error).__name__, error)
    return f"the model failed ({type(error).__name__}); see the server log for details"
