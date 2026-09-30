"""Upstream LLM clients: real OpenAI-compatible endpoint, or a simulated one for demos/tests."""
import asyncio
import json
import time
import uuid

import httpx

from .cache import text_of


class UpstreamError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def _est(n_chars: int) -> int:
    return max(1, n_chars // 4)


class OpenAIUpstream:
    def __init__(self, client: httpx.AsyncClient, cfg):
        self.client, self.cfg = client, cfg

    def _h(self, auth):
        return {"Authorization": auth or f"Bearer {self.cfg.upstream_key}", "Content-Type": "application/json"}

    async def chat(self, body, auth):
        r = await self.client.post(f"{self.cfg.upstream_base}/chat/completions", json={**body, "stream": False}, headers=self._h(auth))
        if r.status_code >= 400:
            raise UpstreamError(r.status_code, r.text)
        return r.json()

    async def stream(self, body, auth):
        b = {**body, "stream": True, "stream_options": {"include_usage": True}}
        async with self.client.stream("POST", f"{self.cfg.upstream_base}/chat/completions", json=b, headers=self._h(auth)) as r:
            if r.status_code >= 400:
                await r.aread()
                raise UpstreamError(r.status_code, r.text)
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                yield json.loads(d)


class MockUpstream:
    """Deterministic fake LLM with realistic latency so cache speed-ups are visible."""

    def __init__(self, cfg):
        self.cfg = cfg

    def _answer(self, body):
        q = text_of(next((m["content"] for m in reversed(body["messages"]) if m["role"] == "user"), ""))
        return (f"[simulated {body.get('model')}] Here is a detailed answer to: \"{q.strip()[:120]}\". "
                "Semantic caching stores the response keyed by meaning, so paraphrased questions reuse it "
                "instead of paying for another completion.")

    def _usage(self, body, ans):
        p = _est(sum(len(text_of(m.get("content"))) for m in body["messages"]))
        return {"prompt_tokens": p, "completion_tokens": _est(len(ans)), "total_tokens": p + _est(len(ans))}

    async def chat(self, body, auth):
        await asyncio.sleep(self.cfg.mock_delay)
        ans = self._answer(body)
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:20]}", "object": "chat.completion", "created": int(time.time()),
                "model": body.get("model"), "choices": [{"index": 0, "message": {"role": "assistant", "content": ans}, "finish_reason": "stop"}],
                "usage": self._usage(body, ans)}

    async def stream(self, body, auth):
        await asyncio.sleep(self.cfg.mock_delay * 0.4)
        ans, cid, ts = self._answer(body), f"chatcmpl-{uuid.uuid4().hex[:20]}", int(time.time())
        base = {"id": cid, "object": "chat.completion.chunk", "created": ts, "model": body.get("model")}
        yield {**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
        words = ans.split(" ")
        for i, w in enumerate(words):
            await asyncio.sleep(self.cfg.mock_delay * 0.6 / len(words))
            yield {**base, "choices": [{"index": 0, "delta": {"content": w + (" " if i < len(words) - 1 else "")}, "finish_reason": None}]}
        yield {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        yield {**base, "choices": [], "usage": self._usage(body, ans)}


def assemble(chunks: list[dict], body: dict) -> dict:
    """Rebuild a full chat.completion object from streamed chunks so it can be cached."""
    content, finish, usage, head = [], "stop", None, chunks[0] if chunks else {}
    for c in chunks:
        for ch in c.get("choices", []):
            content.append(ch.get("delta", {}).get("content") or "")
            finish = ch.get("finish_reason") or finish
        usage = c.get("usage") or usage
    text = "".join(content)
    if not usage:
        p = _est(sum(len(text_of(m.get("content"))) for m in body["messages"]))
        usage = {"prompt_tokens": p, "completion_tokens": _est(len(text)), "total_tokens": p + _est(len(text))}
    return {"id": head.get("id", f"chatcmpl-{uuid.uuid4().hex[:20]}"), "object": "chat.completion",
            "created": head.get("created", int(time.time())), "model": head.get("model", body.get("model")),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}], "usage": usage}


def build_upstream(cfg, client):
    return MockUpstream(cfg) if cfg.mock else OpenAIUpstream(client, cfg)
