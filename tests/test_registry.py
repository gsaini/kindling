import pytest

from kindling.config import Settings
from kindling.llm import ModelNotAllowed, ModelRegistry, UnknownModel
from kindling.llm.ollama import OllamaModel

MODELS = """
default = "nemotron"

[models.nemotron]
name = "nemotron-3-nano:30b"
num_ctx = 16384
aliases = ["nemotron-3-nano"]

[models.nemotron-fast]
name = "nemotron-3-nano:4b"
think = false
"""


def test_models_file_ids_aliases_and_allowlists(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(MODELS)
    registry = ModelRegistry.from_settings(Settings(_env_file=None, models_file=path))
    assert registry.spec().id == "nemotron" and registry.spec("nemotron-3-nano").id == "nemotron"
    spec, model = registry.resolve("nemotron-fast")
    assert spec.think is False and isinstance(model, OllamaModel) and model.name == "nemotron-3-nano:4b"
    assert registry.get("nemotron") is registry.get("nemotron-3-nano")  # built once
    with pytest.raises(UnknownModel):
        registry.spec("gpt-5")
    with pytest.raises(ModelNotAllowed):
        registry.resolve("nemotron", allowed=frozenset({"nemotron-fast"}))


def test_without_a_file_one_model_comes_from_the_environment(tmp_path):
    settings = Settings(_env_file=None, models_file=tmp_path / "none.toml", model="nemotron-3-nano:4b", num_ctx=4096)
    registry = ModelRegistry.from_settings(settings)
    assert [s.id for s in registry.specs()] == ["nemotron-3-nano:4b"] and registry.spec().num_ctx == 4096


def test_a_broken_models_file_says_what_is_wrong(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text('default = "missing"\n[models.a]\nname = "x"\n')
    with pytest.raises(ValueError, match="default model 'missing'"):
        ModelRegistry.from_file(path)
