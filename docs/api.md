# Standard APIs: OpenAI and Anthropic compatible

Kindling speaks the two APIs most tools already know, so the OpenAI and Anthropic SDKs, Claude Code,
and anything else that takes a base URL can use Nemotron through Kindling with a Kindling API key.

These endpoints are **plain model access**, like any OpenAI-compatible server: the `tools` a client
sends are its own tools, returned to it as tool calls to run. Kindling's server-side tools, its
`KINDLING_SYSTEM_PROMPT` and saved conversations belong to `/api/chat` and are not used here.

| Endpoint | API | Notes |
|---|---|---|
| `GET /v1/models` | both | The models this key may use. Anthropic's shape when the request has an `anthropic-version` header. |
| `GET /v1/models/{id}` | both | An id or alias from models.toml. |
| `POST /v1/chat/completions` | OpenAI | Streaming and not. |
| `POST /v1/messages` | Anthropic | Streaming and not. |
| `POST /v1/messages/count_tokens` | Anthropic | An estimate (see below). |

Every call needs a key with the `chat` scope, as `Authorization: Bearer <key>` or `x-api-key: <key>`.
`model` is a model id or alias from models.toml (`nemotron`, `nemotron-fast`...). Each reply carries an
`X-Kindling-Model` header naming the model that actually served it.

## Examples

```bash
export KINDLING=http://localhost:8000 KEY=<your kindling key>

curl $KINDLING/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"model": "nemotron", "messages": [{"role": "user", "content": "Say hi"}]}'

curl $KINDLING/v1/messages -H "x-api-key: $KEY" -H 'anthropic-version: 2023-06-01' -H 'content-type: application/json' \
  -d '{"model": "nemotron", "max_tokens": 1024, "stream": true, "messages": [{"role": "user", "content": "Say hi"}]}'
```

OpenAI SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="<kindling key>")
reply = client.chat.completions.create(
    model="nemotron",
    messages=[{"role": "user", "content": "Why is the sky blue? One sentence."}],
    reasoning_effort="low",  # "none" turns reasoning off
)
print(reply.choices[0].message.model_extra.get("reasoning_content"))  # the model's reasoning
print(reply.choices[0].message.content)

for chunk in client.chat.completions.create(model="nemotron", messages=[...], stream=True):
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

Anthropic SDK:

```python
import anthropic

client = anthropic.Anthropic(base_url="http://localhost:8000", api_key="<kindling key>")  # no /v1
with client.messages.stream(
    model="nemotron",
    max_tokens=4096,
    thinking={"type": "enabled", "budget_tokens": 2048},  # the budget is ignored: reasoning is on or off
    messages=[{"role": "user", "content": "Why is the sky blue? One sentence."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="")
```

## Claude Code

Create a key for it (`kindling keys create claude-code`, see docs/auth.md), then:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000      # or your tunnel URL; no /v1
export ANTHROPIC_AUTH_TOKEN=<kindling key>           # sent as Authorization: Bearer
export ANTHROPIC_MODEL=nemotron                      # the main model
export ANTHROPIC_DEFAULT_HAIKU_MODEL=nemotron-fast   # background work (titles, summaries); optional
claude
```

Claude Code asks for Claude models by name for some of its work (and when `ANTHROPIC_MODEL` isn't
set). **An unknown model name that starts with `claude-` is served by the default model**, and the
reply echoes the name that was asked for, so Claude Code is satisfied; any other unknown name is a
404. Set the variables above to choose the models explicitly.

Claude Code's system prompt and tool definitions are large (well over 16k tokens). A prompt that is
certainly longer than the model's `num_ctx` is refused with `prompt is too long: N tokens > M maximum`
(OpenAI API: 400 `context_length_exceeded`); the check counts six characters per token, so it only
refuses what can't fit. Below that, what happens depends on the runner: Ollama's llama engine drops the
oldest text silently, its MLX engine reads it all, slowly. Either way, give the model a bigger window in
models.toml:

```toml
[models.nemotron-code]
provider = "ollama"
name = "nemotron-3-nano:30b"
num_ctx = 65536
description = "Nemotron 3 Nano with room for Claude Code"
```

Image input (screenshots, image files read by Claude Code) is rejected with a 400 for now.

## Other OpenAI clients

Anything that takes an OpenAI base URL works the same way: base URL `http://<host>:8000/v1`, API key
= a Kindling key, model = an id from models.toml. That covers the OpenAI SDKs in any language,
LangChain / LlamaIndex (`ChatOpenAI(base_url=..., api_key=...)`), Open WebUI, Continue, Aider,
`llm`, and so on. Reasoning arrives as `reasoning_content`, which clients built for vLLM or DeepSeek
show; others ignore it.

## Field mapping

What each request field becomes for the model (Ollama's chat API; the openai provider translates on).

| OpenAI request | Anthropic request | Model call |
|---|---|---|
| `messages` (system, developer, user, assistant, tool) | `system` + `messages` | `messages` in Ollama format; content arrays are joined as text |
| `tool` message `tool_call_id` | `tool_result.tool_use_id` | `tool_name` of the matching earlier call; `is_error` prefixes "Error: " |
| `tools` (function) | `tools` with `input_schema` | `tools`; Anthropic tools without `input_schema` (server tools) are left out |
| `tool_choice` none / auto | `tool_choice` none / auto | no tools / tools |
| `tool_choice` required / named function | `tool_choice` any / tool | best effort: the tool list is narrowed and the system prompt says to call it |
| `reasoning_effort` none, minimal / low, medium, high | `thinking` disabled / enabled | `think` false / true; absent (or `adaptive`): the model's `think` default |
| `think` (Kindling extra body field) | | `think`, winning over `reasoning_effort` |
| `response_format` json_object / json_schema | `output_config.format` json_schema | `format` "json" / the schema |
| `max_tokens`, `max_completion_tokens` | `max_tokens` (required) | `num_predict` |
| `temperature`, `top_p`, `seed`, `presence_penalty`, `frequency_penalty` | `temperature`, `top_p`, `top_k` | the same options |
| `stop` | `stop_sequences` | OpenAI: `stop`. Anthropic: matched by Kindling (below) |

| Reply | OpenAI | Anthropic |
|---|---|---|
| reasoning | `reasoning_content` on `message` / `delta` | `thinking` blocks, first, with `signature: ""` |
| tool calls | `tool_calls`, ids `call_…`, `arguments` a JSON string | `tool_use` blocks, ids `toolu_…`; streamed as `input_json_delta` |
| why it stopped | `finish_reason` stop / tool_calls / length | `stop_reason` end_turn / tool_use / max_tokens / stop_sequence |
| usage | `prompt_tokens`, `completion_tokens`, `total_tokens`; streamed as a last chunk with `stream_options.include_usage` | `input_tokens`, `output_tokens` (in `message_delta` when streaming) |

"length" / "max_tokens" means the output reached the limit; output tokens include reasoning tokens.

Anthropic streams follow the official sequence: `message_start`, `ping`, then per block
`content_block_start` / `content_block_delta` / `content_block_stop`, then `message_delta` and
`message_stop`. `message_start` carries an estimate of `input_tokens`; `message_delta` has the real count.

## Behaviour worth knowing

- **Errors** use each API's own shape on every `/v1/` path: OpenAI
  `{"error": {"message", "type", "param", "code"}}`, Anthropic `{"type": "error", "error": {"type", "message"}}`.
  Validation errors are 400 (not FastAPI's 422); auth errors keep their 401 / 403 / 429 status and
  headers (`WWW-Authenticate`, `Retry-After`). `/api/*` is unchanged.
- **A busy server**: when the generation queue is full, OpenAI clients get a 503 and Anthropic clients
  a 529 `overloaded_error`; both SDKs retry these on their own.
- **Streaming** sends headers within a second: it waits that long for the model's first event, so a
  call that fails at once (a full queue, Ollama down) still gets a real error status. A failure after
  that arrives as an error event in the stream (`data: {"error": ...}` for OpenAI, `event: error` for
  Anthropic), which the SDKs raise.
- **Keepalives**: Cloudflare cuts a request after about 125 s without a byte, and the model can be
  silent longer than that (loading, waiting in the generation queue, reasoning before any output).
  After 15 s of silence, and every 15 s after, a stream gets an SSE comment `: keepalive` (OpenAI;
  clients skip comments) or an `event: ping` (Anthropic, part of its protocol). The helper is
  `kindling.api.compat.keepalive.with_keepalive`.
- **Disconnects**: a client that goes away (streaming or not) stops the generation; its usage is
  recorded with status "stopped".
- **Usage** is recorded per request with endpoint `/v1/chat/completions` or `/v1/messages`.
- **Stop sequences (Anthropic)** are matched by Kindling in the streamed text rather than by the model,
  so the reply can report `stop_reason: "stop_sequence"` and which one fired (Ollama strips the match
  and reports a plain stop). Generation stops at the match; the token counts for such a reply are
  estimates, since the model's own counts never arrive.
- **count_tokens** is an estimate: about four characters per token over the text the model would be
  sent (system, messages, tool calls and results, tool definitions), plus 4 tokens per message for
  the chat template. Ollama has no tokenize endpoint; expect it to be within roughly 20% for English
  and code.

## Upstream: a bigger Nemotron on another machine

A models.toml entry with `provider = "openai"` puts any OpenAI-compatible server (vLLM, NVIDIA NIM,
llama.cpp, LM Studio) behind Kindling; clients keep using Kindling's keys and model ids.

```toml
[models.nemotron-super]
provider = "openai"
name = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"  # as the server's /models lists it
host = "http://gpu-box:8000/v1"                           # base URL, including /v1
api_key_env = "GPU_BOX_API_KEY"                           # optional: sent as a Bearer token
num_ctx = 131072
```

- Requests stream from `{host}/chat/completions` with `stream_options.include_usage`.
- `think` is sent as `chat_template_kwargs: {"enable_thinking": think}`, the switch vLLM uses for
  Nemotron's chat template; servers that don't know it ignore it. (Older NIM containers switch
  Nemotron's reasoning with a "detailed thinking on/off" system prompt instead; add that to your
  messages if yours does.)
- Reasoning is read from `reasoning_content` or `reasoning`; run vLLM with its reasoning parser
  (`--reasoning-parser`, plus `--enable-auto-tool-choice --tool-call-parser ...` for tools) so it
  separates reasoning and tool calls from the text.
- Options map to `max_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `seed`, `stop`,
  `presence_penalty`, `frequency_penalty`, `repetition_penalty`; `num_ctx` is the server's business.
- `kindling` health checks call `GET {host}/models` and confirm the model is listed.

## Known gaps versus the real APIs

- Text only: image, audio, file and document inputs are a 400 "not supported yet".
- One choice per request (`n` > 1 is a 400); no `logprobs`, `logit_bias`, `parallel_tool_calls`
  control, `prediction`, audio output, or the Responses / Assistants / Embeddings APIs.
- `tool_choice` required / any / named tool can't be enforced on Ollama: it is a strong hint, not a
  guarantee. `disable_parallel_tool_use` is ignored. OpenAI `strict` tool schemas are not enforced.
- Thinking `budget_tokens` and `output_config.effort` are ignored (Nemotron's reasoning is on or off).
  Thinking blocks have an empty signature, and thinking sent back in later turns is dropped, as
  Kindling's own chat does.
- No prompt caching: `cache_control` is accepted and ignored, and usage has no cache fields.
- Anthropic server tools (web search, code execution, ...) and MCP connectors are left out of the tool list.
- No Message Batches or Files APIs; `metadata`, `service_tier`, `user` and similar fields are accepted
  and ignored.
- Token counts for `count_tokens` and for replies cut at a stop sequence are estimates.
- Tool call arguments are not streamed piece by piece: each call arrives whole in one delta, because
  Ollama returns tool calls whole.
