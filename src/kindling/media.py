"""Images and audio in user messages: parse the content parts, check them, prepare the audio.

Both chat endpoints take OpenAI's content parts (see docs/api.md):

    {"type": "text", "text": "..."}
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}, "name": "rx.jpg"}
    {"type": "input_audio", "input_audio": {"data": "<base64>", "format": "ogg"}, "name": "voice.ogg"}

A parsed user message keeps its text in `content` (the text parts joined by newlines) and its media in
`media`, a list of dicts: kind ("image" | "audio"), mime, bytes, name, size, seconds (audio only).
The upstream client turns them back into content parts (llm/openai_upstream.py).

Kindling never fetches a URL for a caller: media must be embedded. Audio is converted with ffmpeg to
the 16 kHz mono WAV the model reads, because the model server decodes only WAV, MP3 and FLAC from
memory, and WhatsApp voice notes are OGG/Opus and browsers record WebM or MP4. Without ffmpeg only
WAV is accepted.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import io
import os
import shutil
import wave
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .api.compat.errors import InvalidRequest

if TYPE_CHECKING:
    from .config import Settings
    from .llm import Message, ModelSpec

# Declared type -> how its data starts. WebP is RIFF....WEBP, checked separately.
_IMAGE_MAGIC = {"image/png": b"\x89PNG\r\n\x1a\n", "image/jpeg": b"\xff\xd8\xff", "image/gif": b"GIF8"}
IMAGE_TYPES = (*_IMAGE_MAGIC, "image/webp")
AUDIO_FORMATS = ("wav", "mp3", "flac", "ogg", "opus", "webm", "m4a", "mp4", "aac")
_AUDIO_ALIASES = {"mpeg": "mp3", "x-wav": "wav", "wave": "wav", "x-m4a": "m4a", "x-flac": "flac"}
_WAV_BYTES_PER_SECOND = 16_000 * 2  # 16 kHz, mono, 16-bit
_CONVERT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class MediaLimits:
    max_image_bytes: int = 8_000_000
    max_audio_bytes: int = 15_000_000
    max_audio_seconds: float = 300.0
    max_images: int = 16
    max_audio: int = 8

    @classmethod
    def from_settings(cls, settings: Settings) -> MediaLimits:
        return cls(
            max_image_bytes=settings.media_max_image_bytes,
            max_audio_bytes=settings.media_max_audio_bytes,
            max_audio_seconds=settings.media_max_audio_seconds,
            max_images=settings.media_max_images,
            max_audio=settings.media_max_audio,
        )


def _mb(n: int) -> str:
    return f"{n / 1_000_000:g} MB"


def _decode(data: Any, param: str) -> bytes:
    if not isinstance(data, str) or not data:
        raise InvalidRequest("media data must be a non-empty base64 string", param)
    try:
        return base64.b64decode("".join(data.split()), validate=True)
    except (binascii.Error, ValueError):
        raise InvalidRequest("media data is not valid base64", param) from None


def _image(part: dict[str, Any], param: str, limits: MediaLimits) -> dict[str, Any]:
    source = part.get("image_url")
    url = source.get("url") if isinstance(source, dict) else source
    if not isinstance(url, str) or not url:
        raise InvalidRequest("an image part needs image_url.url", param)
    if not url.startswith("data:"):
        raise InvalidRequest("Kindling does not fetch URLs: send the image as a data: URL (base64)", param)
    header, _, data = url[5:].partition(",")
    mime, _, encoding = header.partition(";")
    mime = mime.strip().lower()
    if encoding.strip().lower() != "base64":
        raise InvalidRequest("an image data: URL must be base64-encoded", param)
    if mime not in IMAGE_TYPES:
        raise InvalidRequest(f"image type {mime or 'unknown'!r} is not supported: send PNG, JPEG, WebP or GIF", param)
    raw = _decode(data, param)
    if len(raw) > limits.max_image_bytes:
        raise InvalidRequest(f"an image is larger than the {_mb(limits.max_image_bytes)} limit", param)
    is_webp = raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"
    if (mime == "image/webp" and not is_webp) or (mime != "image/webp" and not raw.startswith(_IMAGE_MAGIC[mime])):
        raise InvalidRequest(f"the image data is not {mime}", param)
    return {"kind": "image", "mime": mime, "bytes": raw, "name": _name(part), "size": len(raw), "seconds": None}


def _audio(part: dict[str, Any], param: str, limits: MediaLimits) -> dict[str, Any]:
    source = part.get("input_audio")
    if not isinstance(source, dict):
        raise InvalidRequest("an audio part needs input_audio: {data, format}", param)
    data = source.get("data")
    if isinstance(data, str) and data.startswith("data:"):
        data = data.partition(",")[2]
    fmt = str(source.get("format") or "").strip().lower().removeprefix("audio/")
    fmt = _AUDIO_ALIASES.get(fmt, fmt)
    if fmt not in AUDIO_FORMATS:
        raise InvalidRequest(
            f"audio format {fmt or 'missing'!r} is not supported: use one of {', '.join(AUDIO_FORMATS)}", param
        )
    raw = _decode(data, param)
    if len(raw) > limits.max_audio_bytes:
        raise InvalidRequest(f"an audio clip is larger than the {_mb(limits.max_audio_bytes)} limit", param)
    return {
        "kind": "audio",
        "mime": f"audio/{fmt}",
        "format": fmt,
        "bytes": raw,
        "name": _name(part),
        "size": len(raw),
        "seconds": None,
    }


def _name(part: dict[str, Any]) -> str | None:
    name = part.get("name")
    return str(name)[:200] if isinstance(name, str) and name.strip() else None


def split_content(
    content: Any, param: str, *, media_allowed: bool, limits: MediaLimits
) -> tuple[str, list[dict[str, Any]]]:
    """Message content -> (text, media). A string is text; a list is content parts. Media in a
    message that may not carry it (anything but a user message) is refused, never dropped."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        raise InvalidRequest("content must be a string or an array of content parts", param)
    texts: list[str] = []
    media: list[dict[str, Any]] = []
    for i, part in enumerate(content):
        kind = part.get("type") if isinstance(part, dict) else None
        where = f"{param}.{i}"
        if kind in ("text", "refusal"):
            texts.append(str(part.get(kind) or ""))
        elif kind in ("image_url", "input_image", "input_audio"):
            if not media_allowed:
                raise InvalidRequest("images and audio are accepted only in user messages", where)
            media.append(_image(part, where, limits) if kind != "input_audio" else _audio(part, where, limits))
        elif kind == "file":
            raise InvalidRequest("file content is not supported: send text, images or audio", where)
        else:
            raise InvalidRequest(f"unsupported content part type {kind!r}", where)
    return "\n".join(texts), media


def has_media(message: Message) -> bool:
    return bool(message.get("media"))


def newest_user_has_media(messages: Sequence[Message]) -> bool:
    for message in reversed(messages):
        if message.get("role") == "user":
            return has_media(message)
    return False


def think_default(requested: bool | None, model_default: bool | None, messages: Sequence[Message]) -> bool | None:
    """The request's choice wins; otherwise media in the newest user message means reason (questions
    that combine a photo and a voice note were answered wrongly without it); otherwise the model's."""
    if requested is not None:
        return requested
    if newest_user_has_media(messages):
        return True
    return model_default


# -- audio conversion ---------------------------------------------------------------------------


def find_ffmpeg() -> str | None:
    return os.environ.get("KINDLING_FFMPEG") or shutil.which("ffmpeg")


class _ConversionCache:
    """Converted clips by the hash of their input. Clients resend earlier voice notes with every
    turn, so each is converted once. Bounded by total size, oldest out first."""

    def __init__(self, max_bytes: int = 256_000_000):
        self.max_bytes = max_bytes
        self._items: OrderedDict[str, tuple[bytes, float]] = OrderedDict()
        self._size = 0

    def get(self, key: str) -> tuple[bytes, float] | None:
        item = self._items.get(key)
        if item is not None:
            self._items.move_to_end(key)
        return item

    def put(self, key: str, value: tuple[bytes, float]) -> None:
        if key in self._items or len(value[0]) > self.max_bytes:
            return
        self._items[key] = value
        self._size += len(value[0])
        while self._size > self.max_bytes:
            _, (old, _) = self._items.popitem(last=False)
            self._size -= len(old)


_cache = _ConversionCache()


def _wav_seconds(raw: bytes) -> float:
    with wave.open(io.BytesIO(raw)) as w:
        return w.getnframes() / float(w.getframerate() or 1)


def _fix_wav_header(raw: bytes) -> bytes:
    """ffmpeg writing WAV to a pipe can't seek back to fill in the sizes; fill them in here so every
    reader sees the right length."""
    at = raw.find(b"data", 12)
    if raw[:4] != b"RIFF" or at < 0:
        return raw
    out = bytearray(raw)
    out[4:8] = (len(raw) - 8).to_bytes(4, "little")
    out[at + 4 : at + 8] = (len(raw) - at - 8).to_bytes(4, "little")
    return bytes(out)


async def _ffmpeg_to_wav(ffmpeg: str, raw: bytes, max_seconds: float) -> bytes:
    process = await asyncio.create_subprocess_exec(
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", "-f", "wav", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )  # fmt: skip
    try:
        out, _ = await asyncio.wait_for(process.communicate(raw), _CONVERT_TIMEOUT_SECONDS)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    if process.returncode != 0 or len(out) <= 44:
        raise ValueError("ffmpeg could not read it")
    # A little slack over the limit, so the limit check below names seconds rather than bytes.
    if len(out) > (max_seconds + 5) * _WAV_BYTES_PER_SECOND + 1024:
        raise OverflowError
    return _fix_wav_header(out)


async def _prepare_audio(item: dict[str, Any], param: str, limits: MediaLimits, ffmpeg: str | None) -> None:
    raw: bytes = item["bytes"]
    fmt: str = item["format"]
    if ffmpeg is None:
        if fmt != "wav":
            raise InvalidRequest(f"this server cannot convert {fmt} audio (ffmpeg is not installed): send WAV", param)
        try:
            seconds = _wav_seconds(raw)
        except (wave.Error, EOFError):
            raise InvalidRequest("the audio is not readable WAV", param) from None
        wav = raw
    else:
        key = hashlib.sha256(raw).hexdigest()
        cached = _cache.get(key)
        if cached is None:
            try:
                wav = await _ffmpeg_to_wav(ffmpeg, raw, limits.max_audio_seconds)
            except OverflowError:
                raise InvalidRequest(
                    f"an audio clip is longer than the {limits.max_audio_seconds:g} second limit", param
                ) from None
            except (ValueError, TimeoutError):
                raise InvalidRequest(f"the audio is not readable {fmt} audio", param) from None
            seconds = (len(wav) - wav.find(b"data", 12) - 8) / _WAV_BYTES_PER_SECOND
            _cache.put(key, (wav, seconds))
        else:
            wav, seconds = cached
    if seconds > limits.max_audio_seconds:
        raise InvalidRequest(f"an audio clip is longer than the {limits.max_audio_seconds:g} second limit", param)
    item.update(bytes=wav, mime="audio/wav", format="wav", seconds=round(seconds, 2))


async def prepare(
    messages: Sequence[Message],
    spec: ModelSpec,
    limits: MediaLimits,
    *,
    ffmpeg: str | None = None,
    convert: bool = True,
) -> None:
    """Check the media of a whole request against the model and the limits, and convert its audio,
    in place. `param` paths name the message the problem is in."""
    images = audio = 0
    for message in messages:
        for item in message.get("media") or []:
            if item["kind"] == "image":
                images += 1
            else:
                audio += 1
    if images and "image" not in spec.inputs:
        raise InvalidRequest(f"model {spec.id!r} does not accept images", "messages")
    if audio and "audio" not in spec.inputs:
        raise InvalidRequest(f"model {spec.id!r} does not accept audio", "messages")
    if images > limits.max_images:
        raise InvalidRequest(
            f"a request may carry at most {limits.max_images} images; this one has {images}", "messages"
        )
    if audio > limits.max_audio:
        raise InvalidRequest(
            f"a request may carry at most {limits.max_audio} audio clips; this one has {audio}", "messages"
        )
    if not (audio and convert):
        return
    for i, message in enumerate(messages):
        for j, item in enumerate(message.get("media") or []):
            if item["kind"] == "audio" and item.get("seconds") is None:
                await _prepare_audio(item, f"messages.{i}.content (media {j})", limits, ffmpeg)


def to_part(item: dict[str, Any]) -> dict[str, Any]:
    """One media item as the OpenAI content part a model server reads."""
    data = base64.b64encode(item["bytes"]).decode()
    if item["kind"] == "image":
        return {"type": "image_url", "image_url": {"url": f"data:{item['mime']};base64,{data}"}}
    return {"type": "input_audio", "input_audio": {"data": data, "format": item.get("format") or "wav"}}
