"""The models this server offers, from a per-machine `models.toml`.

Clients ask for a model by its id or an alias ("nemotron"); the registry maps that to a provider
and an upstream model ("nemotron-3-nano:30b" on Ollama here, a bigger Nemotron on a GPU box).

    default = "nemotron"

    [models.nemotron]
    provider = "ollama"
    name = "nemotron-3-nano:30b"
    num_ctx = 16384
    aliases = ["nemotron-3-nano"]

Without a models file, one model is built from KINDLING_MODEL / KINDLING_NUM_CTX / KINDLING_OLLAMA_HOST.
"""

from __future__ import annotations

import tomllib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from .base import ChatModel

if TYPE_CHECKING:
    from ..config import Settings


class ModelSpec(BaseModel):
    id: str  # what clients send as `model`
    provider: Literal["ollama", "openai"] = "ollama"
    name: str  # the upstream model name
    host: str | None = None  # Ollama host, or the base URL of an OpenAI-compatible server (vLLM, NIM, llama.cpp)
    api_key_env: str | None = None  # env var with the upstream key, for the openai provider
    num_ctx: int = 16384
    think: bool | None = None  # reasoning default when a request doesn't say; None: the model's own default
    tools: bool = True  # offer the server's tools to this model
    description: str = ""
    aliases: list[str] = Field(default_factory=list)


class ModelsFile(BaseModel):
    default: str
    models: dict[str, dict[str, Any]]

    @model_validator(mode="after")
    def default_exists(self) -> ModelsFile:
        if self.default not in self.models:
            raise ValueError(f"default model {self.default!r} is not defined under [models]")
        return self


class UnknownModel(LookupError):
    pass


class ModelNotAllowed(PermissionError):
    pass


Wrapper = Callable[[ChatModel], ChatModel]


def build_model(spec: ModelSpec) -> ChatModel:
    if spec.provider == "ollama":
        from .ollama import OllamaModel

        model: ChatModel = OllamaModel(spec.name, host=spec.host, num_ctx=spec.num_ctx)
    else:
        from .openai_upstream import OpenAICompatModel

        model = OpenAICompatModel(spec)
    return model


class ModelRegistry:
    def __init__(
        self,
        specs: Iterable[ModelSpec],
        default: str,
        *,
        models: dict[str, ChatModel] | None = None,
        wrap: Wrapper | None = None,
    ):
        self._specs = {s.id: s for s in specs}
        if default not in self._specs:
            raise ValueError(f"default model {default!r} is not defined")
        self.default = default
        self._aliases: dict[str, str] = {}
        for spec in self._specs.values():
            for alias in [spec.id, *spec.aliases]:
                if alias in self._aliases and self._aliases[alias] != spec.id:
                    raise ValueError(f"model name {alias!r} is used twice")
                self._aliases[alias] = spec.id
        self._wrap = wrap
        self._models: dict[str, ChatModel] = {}
        for model_id, model in (models or {}).items():
            self._models[model_id] = wrap(model) if wrap else model

    # -- construction -----------------------------------------------------------------------

    @classmethod
    def single(cls, model: ChatModel, *, think: bool | None = None, wrap: Wrapper | None = None) -> ModelRegistry:
        """One already-built model (tests, scripts)."""
        spec = ModelSpec(id=model.name, name=model.name, num_ctx=model.num_ctx, think=think)
        return cls([spec], spec.id, models={spec.id: model}, wrap=wrap)

    @classmethod
    def from_file(cls, path: Path, *, wrap: Wrapper | None = None) -> ModelRegistry:
        try:
            raw = ModelsFile.model_validate(tomllib.loads(path.read_text()))
            specs = [ModelSpec(id=model_id, **fields) for model_id, fields in raw.models.items()]
        except (tomllib.TOMLDecodeError, ValidationError) as error:
            raise ValueError(f"{path} is not a valid models file: {error}") from error
        return cls(specs, raw.default, wrap=wrap)

    @classmethod
    def from_settings(cls, settings: Settings, *, wrap: Wrapper | None = None) -> ModelRegistry:
        if settings.models_file.is_file():
            return cls.from_file(settings.models_file, wrap=wrap)
        spec = ModelSpec(id=settings.model, name=settings.model, host=settings.ollama_host, num_ctx=settings.num_ctx)
        return cls([spec], spec.id, wrap=wrap)

    # -- lookup -----------------------------------------------------------------------------

    def specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def spec(self, model_id: str | None = None) -> ModelSpec:
        key = self._aliases.get(model_id or self.default)
        if key is None:
            raise UnknownModel(f"no model named {model_id!r}. Available: {', '.join(self._specs)}")
        return self._specs[key]

    def get(self, model_id: str | None = None) -> ChatModel:
        spec = self.spec(model_id)
        if spec.id not in self._models:
            model = build_model(spec)
            self._models[spec.id] = self._wrap(model) if self._wrap else model
        return self._models[spec.id]

    def resolve(self, model_id: str | None, allowed: frozenset[str] | None = None) -> tuple[ModelSpec, ChatModel]:
        """The spec and model for a request; `allowed` is the caller's model allowlist (None: any)."""
        spec = self.spec(model_id)
        if allowed is not None and spec.id not in allowed:
            raise ModelNotAllowed(f"this key may not use the model {spec.id!r}")
        return spec, self.get(spec.id)
