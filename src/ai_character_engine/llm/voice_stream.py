"""Measured SSE collection behind the existing LLMClient / gateway contract.

This measures real provider text deltas but buffers them for CharacterRuntime.
It does not claim token-to-speaker streaming or count reasoning/tool deltas as text.
"""
from __future__ import annotations

import asyncio
import time
import re
from types import SimpleNamespace as NS

from .local import OpenAICompatibleChatClient
from .errors import LLMError
from .models import LLMStreamChunk


class VoiceStreamingChatClient(OpenAICompatibleChatClient):
    def __init__(self, *, extra_body=None, max_tokens=512, require_time_tool=False, **kwargs):
        request_options = dict(kwargs.pop("request_options", {}) or {})
        request_options.setdefault("max_tokens", max_tokens)
        if extra_body:
            request_options.setdefault("extra_body", extra_body)
        super().__init__(request_options=request_options, **kwargs)
        self.extra_body = extra_body or {}
        self.max_tokens = max_tokens
        self.require_time_tool = require_time_tool

    async def stream_generate(self, messages, *, tools=None):
        force_clock = (
            self.require_time_tool
            and tools
            and messages
            and messages[-1].role == "user"
            and any(t.name == "get_current_time" for t in tools)
            and re.search(
                r"(現在|现在).*(幾點|几点|時間|时间)|what time|current time|time now",
                messages[-1].content,
                re.IGNORECASE,
            )
        )
        if force_clock:
            response = await self.generate(messages, tools=tools)
            if response.text:
                yield LLMStreamChunk(text=response.text)
            yield LLMStreamChunk(final=True, response=response)
            return
        async for update in super().stream_generate(messages, tools=tools):
            yield update

    async def _call(self, messages, *, tools):
        async def collect():
            started = time.perf_counter()
            request = dict(model=self.model, messages=self._to_chat_messages(messages),
                           stream=True, max_tokens=self.max_tokens)
            if tools:
                request['tools'] = [self._to_chat_tool(t) for t in tools]
            force_clock = (self.require_time_tool and tools and messages
                and messages[-1].role == 'user'
                and any(t.name == 'get_current_time' for t in tools)
                and re.search(r'(現在|现在).*(幾點|几点|時間|时间)|what time|current time|time now',
                              messages[-1].content, re.IGNORECASE))
            if force_clock:
                request['tools'] = [self._to_chat_tool(t) for t in tools if t.name == 'get_current_time']
                request['tool_choice'] = 'required'
                request['messages'].append({'role': 'system', 'content':
                    'For this turn, call get_current_time now with {}. Do not answer or invent the time before the tool result. 請立即呼叫 get_current_time 工具。'})
            if self.extra_body:
                request['extra_body'] = self.extra_body
            stream = await self.client.chat.completions.create(**request)
            text, calls, first_text_ms, first_event_ms = [], {}, None, None
            model, usage, finish = self.model, None, None
            size = 0
            try:
                async for chunk in stream:
                    model = getattr(chunk, 'model', None) or model
                    usage = getattr(chunk, 'usage', None) or usage
                    for choice in chunk.choices:
                        if choice.index != 0:
                            continue
                        delta = choice.delta
                        content = getattr(delta, 'content', None)
                        tool_deltas = getattr(delta, 'tool_calls', None) or []
                        elapsed = (time.perf_counter() - started) * 1000
                        if (content or tool_deltas) and first_event_ms is None:
                            first_event_ms = elapsed
                        if content:
                            if first_text_ms is None:
                                first_text_ms = elapsed
                            text.append(content)
                            size += len(content)
                        for raw in tool_deltas:
                            entry = calls.setdefault(raw.index, {'id': '', 'name': '', 'arguments': ''})
                            entry['id'] += getattr(raw, 'id', None) or ''
                            fn = getattr(raw, 'function', None)
                            if fn:
                                for key in ('name', 'arguments'):
                                    value = getattr(fn, key, None) or ''
                                    entry[key] += value
                                    size += len(value)
                        if size > 1_000_000 or len(calls) > 16:
                            raise LLMError('Provider stream exceeded response limits')
                        finish = getattr(choice, 'finish_reason', None) or finish
            finally:
                await stream.close()
            if finish not in ('stop', 'tool_calls'):
                raise LLMError('Incomplete LLM response; increase max tokens or disable reasoning')
            if any(not c['id'] or not c['name'] for c in calls.values()):
                raise LLMError('Incomplete streamed tool call')
            if force_clock and not any(c['name'] == 'get_current_time' for c in calls.values()):
                raise LLMError('Provider ignored required current-time tool call')
            if not text and not calls:
                raise LLMError('LLM returned no text or tool call; check reasoning settings')
            return NS(choices=[NS(message=NS(content=''.join(text), tool_calls=[
                NS(id=c['id'], function=NS(name=c['name'], arguments=c['arguments']))
                for _, c in sorted(calls.items())]))], model=model, usage=usage,
                ttft_ms=first_text_ms, first_event_ms=first_event_ms)
        async with asyncio.timeout(self.timeout_seconds):
            return await collect()

    def _normalize(self, response, *, latency_ms, attempt):
        result = super()._normalize(response, latency_ms=latency_ms, attempt=attempt)
        result.metadata['voice_stream'] = {
            'ttft_ms': response.ttft_ms, 'first_event_ms': response.first_event_ms,
            'measurement': 'first_nonempty_content_delta', 'buffered_for_runtime': True,
        }
        return result


class MeasuredLLM:
    """Wrap the gateway; records all tool rounds for one sequential voice session."""
    def __init__(self, client, *, clock=time.perf_counter):
        self.client, self.clock = client, clock
        self.reset()

    def reset(self):
        self.calls = []

    async def generate(self, messages, *, tools=None):
        started = self.clock()
        response = await self.client.generate(messages, tools=tools)
        self.calls.append({'started': started, 'total_ms': (self.clock() - started) * 1000,
                           'ttft_ms': response.metadata.get('voice_stream', {}).get('ttft_ms'),
                           'tool_calls': len(response.tool_calls)})
        return response

    def metrics(self):
        first = next((c for c in self.calls if c['ttft_ms'] is not None), None)
        return {
            'llm_ttft_ms': ((first['started'] - self.calls[0]['started']) * 1000
                            + first['ttft_ms']) if first else None,
            'llm_total_latency_ms': sum(c['total_ms'] for c in self.calls),
            'llm_rounds': len(self.calls),
            'llm_rounds_detail': [dict(total_ms=c['total_ms'], ttft_ms=c['ttft_ms'],
                                      tool_calls=c['tool_calls']) for c in self.calls],
        }
