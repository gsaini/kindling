"""Shared plumbing for the OpenAI- and Anthropic-compatible APIs (kindling.api.openai_compat, anthropic_compat).

errors     error bodies in each API's own shape, for every /v1/ path
streaming  running a model call: metering, disconnects, SSE responses with heartbeats
"""
