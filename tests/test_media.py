"""Images and audio in user messages: parsing, checks, audio conversion, both endpoints, saved media."""

from __future__ import annotations

import base64
import io
import shutil
import subprocess
import uuid
import wave

import anyio
import pytest

from kindling.api.compat.errors import InvalidRequest
from kindling.llm import FakeModel, ModelRegistry, ModelSpec, Turn
from kindling.llm.openai_upstream import to_openai_messages
from kindling.media import MediaLimits, prepare, split_content
from kindling.store import MemoryStore, NewMedia, NewMessage

from .conftest import ADMIN, read_events

PNG_1PX = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not installed")
LIMITS = MediaLimits()


def wav(seconds: float, rate: int = 16_000) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\0\0" * int(seconds * rate))
    return out.getvalue()


def ogg(seconds: float) -> bytes:
    """A real OGG/Opus clip, as WhatsApp sends voice notes."""
    assert FFMPEG
    args = [FFMPEG, "-loglevel", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}"]
    return subprocess.run([*args, "-c:a", "libopus", "-f", "ogg", "pipe:1"], capture_output=True, check=True).stdout


def image_part(data: str = PNG_1PX, mime: str = "image/png", **extra) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}, **extra}


def audio_part(raw: bytes, fmt: str, **extra) -> dict:
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(raw).decode(), "format": fmt}, **extra}


def omni_spec(**fields) -> ModelSpec:
    return ModelSpec(id="omni", name="omni", provider="openai", inputs=["text", "image", "audio"], **fields)


def omni_client(make_client, *turns: Turn, store: MemoryStore | None = None, **overrides):
    model = FakeModel(*turns)
    registry = ModelRegistry([omni_spec()], "omni", models={"omni": model})
    client = make_client(models=registry, store=store or MemoryStore(), settings_overrides=overrides or None)
    return client, model


# -- parsing and checks ---------------------------------------------------------------------------


def test_text_parts_and_media_are_split():
    text, media = split_content(
        [{"type": "text", "text": "What is this?"}, image_part(name="rx.png")],
        "messages.0.content",
        media_allowed=True,
        limits=LIMITS,
    )
    assert text == "What is this?"
    assert [(m["kind"], m["mime"], m["name"], m["size"]) for m in media] == [("image", "image/png", "rx.png", 70)]
    assert split_content("plain", "p", media_allowed=False, limits=LIMITS) == ("plain", [])


@pytest.mark.parametrize(
    ("part", "message"),
    [
        ({"type": "image_url", "image_url": {"url": "https://example.com/x.png"}}, "does not fetch URLs"),
        (image_part(mime="image/jpeg"), "not image/jpeg"),
        (image_part(mime="image/tiff"), "not supported"),
        (image_part(data="%%%"), "not valid base64"),
        (audio_part(b"x", "aiff"), "audio format 'aiff' is not supported"),
        ({"type": "file", "file": {}}, "file content is not supported"),
    ],
)
def test_bad_media_is_refused_with_the_reason(part, message):
    with pytest.raises(InvalidRequest, match=message):
        split_content([part], "messages.0.content", media_allowed=True, limits=LIMITS)


def test_media_outside_user_messages_and_over_size_limits_is_refused():
    with pytest.raises(InvalidRequest, match="only in user messages"):
        split_content([image_part()], "p", media_allowed=False, limits=LIMITS)
    with pytest.raises(InvalidRequest, match="larger than the 1e-05 MB limit"):
        split_content([image_part()], "p", media_allowed=True, limits=MediaLimits(max_image_bytes=10))


def _messages(*parts) -> list[dict]:
    _, media = split_content(list(parts), "p", media_allowed=True, limits=LIMITS)
    return [{"role": "user", "content": "", "media": media}]


def test_the_model_must_accept_the_media_and_counts_are_limited():
    text_only = ModelSpec(id="plain", name="plain")
    with pytest.raises(InvalidRequest, match="'plain' does not accept images"):
        anyio.run(prepare, _messages(image_part()), text_only, LIMITS)
    with pytest.raises(InvalidRequest, match="at most 1 images; this one has 2"):
        anyio.run(prepare, _messages(image_part(), image_part()), omni_spec(), MediaLimits(max_images=1))


def test_without_ffmpeg_only_wav_is_accepted():
    messages = _messages(audio_part(wav(2.0), "wav"))
    anyio.run(lambda: prepare(messages, omni_spec(), LIMITS, ffmpeg=None))
    assert messages[0]["media"][0]["seconds"] == 2.0
    with pytest.raises(InvalidRequest, match="cannot convert ogg audio .* send WAV"):
        anyio.run(lambda: prepare(_messages(audio_part(b"OggS....", "ogg")), omni_spec(), LIMITS, ffmpeg=None))
    with pytest.raises(InvalidRequest, match="longer than the 1 second limit"):
        anyio.run(
            lambda: prepare(_messages(audio_part(wav(2.0), "wav")), omni_spec(), MediaLimits(max_audio_seconds=1))
        )


@needs_ffmpeg
def test_ogg_voice_notes_become_16k_mono_wav():
    messages = _messages(audio_part(ogg(1.5), "ogg", name="voice.ogg"))
    anyio.run(lambda: prepare(messages, omni_spec(), LIMITS, ffmpeg=FFMPEG))
    item = messages[0]["media"][0]
    assert (item["mime"], item["format"], item["name"]) == ("audio/wav", "wav", "voice.ogg")
    assert 1.4 < item["seconds"] < 1.7
    with wave.open(io.BytesIO(item["bytes"])) as w:
        assert (w.getnchannels(), w.getframerate(), w.getsampwidth()) == (1, 16_000, 2)
        assert abs(w.getnframes() / 16_000 - item["seconds"]) < 0.05  # the header sizes were filled in


@needs_ffmpeg
def test_unreadable_and_too_long_audio_is_refused():
    with pytest.raises(InvalidRequest, match="not readable ogg audio"):
        anyio.run(lambda: prepare(_messages(audio_part(b"not audio" * 50, "ogg")), omni_spec(), LIMITS, ffmpeg=FFMPEG))
    with pytest.raises(InvalidRequest, match="longer than the 1 second limit"):
        anyio.run(
            lambda: prepare(
                _messages(audio_part(ogg(3), "ogg")), omni_spec(), MediaLimits(max_audio_seconds=1), ffmpeg=FFMPEG
            )
        )


def test_media_goes_upstream_as_content_parts_media_first():
    messages = _messages(image_part(), audio_part(wav(0.5), "wav"))
    messages[0]["content"] = "Is the caller the patient?"
    [message] = to_openai_messages(messages)
    kinds = [p["type"] for p in message["content"]]
    assert kinds == ["image_url", "input_audio", "text"]
    assert message["content"][0]["image_url"]["url"] == f"data:image/png;base64,{PNG_1PX}"
    assert message["content"][1]["input_audio"]["format"] == "wav"
    assert to_openai_messages([{"role": "user", "content": "hi"}]) == [{"role": "user", "content": "hi"}]


def test_media_inputs_need_the_openai_provider():
    with pytest.raises(ValueError, match="image input needs provider"):
        ModelSpec(id="x", name="x", provider="ollama", inputs=["text", "image"])
    assert ModelSpec(id="x", name="x").inputs == ["text"]


# -- endpoints ------------------------------------------------------------------------------------


def test_info_lists_inputs(make_client):
    client, _ = omni_client(make_client)
    info = client.get("/api/info", headers=ADMIN).json()
    assert info["inputs"] == ["text", "image", "audio"]
    assert info["models"][0]["inputs"] == ["text", "image", "audio"]
    assert make_client().get("/api/info", headers=ADMIN).json()["inputs"] == ["text"]


def test_api_chat_takes_media_and_reasons_by_default(make_client):
    client, model = omni_client(make_client, Turn(content="Metformin"), Turn(content="Ok"))
    content = [image_part(name="rx.png"), {"type": "text", "text": "List the medicines"}]
    events = read_events(
        client.post("/api/chat", headers=ADMIN, json={"messages": [{"role": "user", "content": content}]})
    )
    assert events[-1]["type"] == "done"
    sent = model.calls[0]["messages"][-1]
    assert sent["content"] == "List the medicines" and sent["media"][0]["kind"] == "image"
    assert model.calls[0]["think"] is True  # media, and the request did not say

    client.post("/api/chat", headers=ADMIN, json={"messages": [{"role": "user", "content": content}], "think": False})
    assert model.calls[1]["think"] is False  # an explicit choice wins


def test_api_chat_refuses_media_it_cannot_use(make_client):
    plain = make_client()
    response = plain.post("/api/chat", headers=ADMIN, json={"messages": [{"role": "user", "content": [image_part()]}]})
    assert response.status_code == 400 and "does not accept images" in response.json()["detail"]

    client, _ = omni_client(make_client)
    bad = [{"role": "assistant", "content": [image_part()]}, {"role": "user", "content": "x"}]
    response = client.post("/api/chat", headers=ADMIN, json={"messages": bad})
    assert response.status_code == 400 and "only in user messages" in response.json()["detail"]


def test_v1_chat_takes_media_and_respects_reasoning_effort(make_client):
    client, model = omni_client(make_client, Turn(content="Hello"), Turn(content="Hello"))
    content = [audio_part(wav(1.0), "wav"), {"type": "text", "text": "Reply to this"}]
    body = {"model": "omni", "messages": [{"role": "user", "content": content}]}
    assert client.post("/v1/chat/completions", headers=ADMIN, json=body).status_code == 200
    assert model.calls[0]["messages"][-1]["media"][0]["seconds"] == 1.0 and model.calls[0]["think"] is True
    assert (
        client.post("/v1/chat/completions", headers=ADMIN, json={**body, "reasoning_effort": "none"}).status_code == 200
    )
    assert model.calls[1]["think"] is False


def test_saved_conversations_keep_media(make_client):
    store = MemoryStore()
    client, _ = omni_client(make_client, Turn(content="A prescription"), store=store)
    message = {"role": "user", "content": [image_part(name="rx.png")]}
    events = read_events(client.post("/api/chat", headers=ADMIN, json={"messages": [message], "save": True}))
    conversation_id = events[0]["id"]
    saved = client.get(f"/api/conversations/{conversation_id}", headers=ADMIN).json()
    assert saved["title"] == "Image"
    [info] = saved["messages"][0]["media"]
    assert (info["kind"], info["mime"], info["name"], info["size"]) == ("image", "image/png", "rx.png", 70)
    assert "data" not in info

    download = client.get(f"/api/conversations/{conversation_id}/media/{info['id']}", headers=ADMIN)
    assert download.status_code == 200 and download.headers["content-type"] == "image/png"
    assert download.content == base64.b64decode(PNG_1PX)
    missing = client.get(f"/api/conversations/{conversation_id}/media/{uuid.uuid4()}", headers=ADMIN)
    assert missing.status_code == 404


def test_media_belongs_to_the_conversations_owner():
    async def scenario():
        store = MemoryStore()
        conversation = await store.create("asha")
        media = NewMedia(kind="image", mime="image/png", data=b"png")
        await store.append("asha", conversation.id, [NewMessage(role="user", media=[media])])
        [message] = (await store.get("asha", conversation.id)).messages
        media_id = message.media[0].id
        assert (await store.get_media("asha", conversation.id, media_id)).data == b"png"
        assert await store.get_media("ravi", conversation.id, media_id) is None
        await store.delete("asha", conversation.id)
        assert await store.get_media("asha", conversation.id, media_id) is None

    anyio.run(scenario)
