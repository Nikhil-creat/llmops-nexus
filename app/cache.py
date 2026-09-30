"""Two-tier semantic cache: L1 exact-hash LRU -> L2 vector similarity, with safety guards."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Awaitable, Callable

VOLATILE = re.compile(
    r"\b(today|tonight|now|latest|currently|current|right now|this (?:week|month|year)|news|weather|stock price|score)\b", re.I
)
NUM = re.compile(r"\d+(?:\.\d+)?")
NEG = re.compile(r"\b(not|no|never|without|don't|dont|cannot|can't|cant)\b", re.I)


@dataclass
class Lookup:
    kind: str  # exact | semantic | miss | bypass
    similarity: float = 0.0
    payload: dict | None = None
    reason: str = ""
    cacheable: bool = True

    @property
    def hit(self) -> bool:
        return self.kind in ("exact", "semantic")


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


class SemanticCache:
    def __init__(self, cfg, embedder, store):
        self.cfg, self.embedder, self.store = cfg, embedder, store
        self.l1: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self.inflight: dict[str, asyncio.Future] = {}
        self._emb: OrderedDict[str, object] = OrderedDict()

    # ---- keys & scoping -------------------------------------------------
    def scope_of(self, body: dict) -> str:
        sys_msgs = [text_of(m.get("content")) for m in body.get("messages", []) if m.get("role") in ("system", "developer")]
        t = body.get("temperature")
        sig = {
            "m": body.get("model"), "s": sys_msgs, "t": None if t is None else round(float(t), 1),
            "p": body.get("top_p"), "rf": body.get("response_format"),
        }
        return hashlib.sha256(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:16]

    def query_text(self, body: dict) -> str:
        turns = [text_of(m.get("content")) for m in body.get("messages", []) if m.get("role") == "user"]
        return "\n".join(turns[-(self.cfg.context_turns + 1):]).strip()

    def exact_key(self, body: dict) -> str:
        norm = [(m.get("role"), " ".join(text_of(m.get("content")).lower().split())) for m in body.get("messages", [])]
        return hashlib.sha256((self.scope_of(body) + json.dumps(norm)).encode()).hexdigest()

    # ---- policy ---------------------------------------------------------
    def check(self, body: dict) -> str | None:
        """Return a reason string if this request must NOT be cached."""
        if body.get("tools") or body.get("functions"):
            return "tools"
        if (body.get("n") or 1) > 1:
            return "n>1"
        t = body.get("temperature")
        if t is not None and float(t) > self.cfg.max_temperature:
            return "temperature"
        for m in body.get("messages", []):
            if isinstance(m.get("content"), list) and any(p.get("type") != "text" for p in m["content"] if isinstance(p, dict)):
                return "multimodal"
        q = self.query_text(body)
        if not q:
            return "empty"
        if VOLATILE.search(q):
            return "time-sensitive"
        return None

    def ttl_for(self, body: dict, override: int | None = None) -> int:
        if override:
            return override
        t = body.get("temperature")
        if t is not None and float(t) == 0:
            return self.cfg.ttl * 4  # deterministic answers live longer
        if t is not None and float(t) > 0.7:
            return max(60, self.cfg.ttl // 4)  # creative answers go stale faster
        return self.cfg.ttl

    @staticmethod
    def _safe(q: str, cached: str) -> bool:
        """Reject look-alike prompts whose numbers or polarity differ ("5 km" vs "50 km", "do" vs "do not")."""
        return set(NUM.findall(q)) == set(NUM.findall(cached)) and bool(NEG.search(q)) == bool(NEG.search(cached))

    async def _embed(self, text: str):
        if text in self._emb:
            self._emb.move_to_end(text)
            return self._emb[text]
        v = await self.embedder.embed(text)
        self._emb[text] = v
        if len(self._emb) > 512:
            self._emb.popitem(last=False)
        return v

    # ---- operations -----------------------------------------------------
    async def lookup(self, body: dict, threshold: float | None = None) -> Lookup:
        reason = self.check(body)
        if reason:
            return Lookup("bypass", reason=reason, cacheable=False)
        now, key = time.time(), self.exact_key(body)
        e = self.l1.get(key)
        if e and e[0] > now:
            self.l1.move_to_end(key)
            return Lookup("exact", 1.0, e[1])
        self.l1.pop(key, None)

        q = self.query_text(body)
        hit = await self.store.search(await self._embed(q), self.scope_of(body))
        if not hit:
            return Lookup("miss")
        thr = threshold or self.cfg.threshold
        if hit.score >= thr and hit.payload.get("expires_at", 1e18) > now and self._safe(q, hit.payload.get("prompt", "")):
            return Lookup("semantic", hit.score, hit.payload)
        return Lookup("miss", hit.score)

    async def put(self, body: dict, response: dict, ttl: int | None = None, latency_ms: float = 0.0) -> None:
        if self.check(body):
            return
        ttl, now, q = self.ttl_for(body, ttl), time.time(), self.query_text(body)
        payload = {"response": response, "prompt": q, "model": body.get("model"), "created": now,
                   "expires_at": now + ttl, "latency_ms": latency_ms}
        self.l1[self.exact_key(body)] = (now + ttl, payload)
        while len(self.l1) > self.cfg.l1_size:
            self.l1.popitem(last=False)
        await self.store.upsert(uuid.uuid4().hex if self.store.name != "qdrant" else str(uuid.uuid4()),
                                await self._embed(q), self.scope_of(body), payload, ttl)

    async def single_flight(self, key: str, factory: Callable[[], Awaitable[dict]]):
        """Coalesce identical in-flight requests: one upstream call, N waiters."""
        fut = self.inflight.get(key)
        if fut:
            return await asyncio.shield(fut), False
        fut = asyncio.get_running_loop().create_future()
        self.inflight[key] = fut
        try:
            res = await factory()
            fut.set_result(res)
            return res, True
        except BaseException as exc:
            fut.set_exception(exc)
            fut.exception()  # mark retrieved
            raise
        finally:
            self.inflight.pop(key, None)

    async def flush(self):
        self.l1.clear()
        await self.store.flush()
