import asyncio
import os
import time

os.environ.update(MOCK_UPSTREAM="1", MOCK_DELAY_S="0.05", VECTOR_STORE="memory", EMBEDDER="local")

from fastapi.testclient import TestClient

from app.cache import SemanticCache
from app.config import Settings
from app.embedder import LocalEmbedder
from app.main import app
from app.stores import MemoryStore


def make():
    cfg = Settings()
    return SemanticCache(cfg, LocalEmbedder(), MemoryStore(384)), cfg


def req(q, system="You are helpful.", **kw):
    return {"model": "gpt-4o-mini", "messages": [{"role": "system", "content": system}, {"role": "user", "content": q}], **kw}


RESP = {"id": "x", "created": 1, "model": "m", "choices": [{"message": {"content": "a"}}], "usage": {"total_tokens": 10}}


def run(c):
    return asyncio.run(c)


def test_exact_and_semantic_hit():
    async def go():
        c, _ = make()
        await c.put(req("How do I reset my password?"), RESP)
        assert (await c.lookup(req("How do I reset my password?"))).kind == "exact"
        assert (await c.lookup(req("how can i reset my password"))).kind == "semantic"
        assert (await c.lookup(req("How do I delete my account?"))).kind == "miss"
    run(go())


def test_scope_isolation():
    async def go():
        c, _ = make()
        await c.put(req("reset my password", system="You are a bank bot."), RESP)
        assert (await c.lookup(req("reset my password", system="You are a pirate."))).kind == "miss"
    run(go())


def test_number_and_negation_guards():
    async def go():
        c, _ = make()
        await c.put(req("convert 5 km to miles"), RESP)
        assert (await c.lookup(req("convert 50 km to miles"))).kind == "miss"
        await c.put(req("should I enable two factor auth"), RESP)
        assert (await c.lookup(req("should I not enable two factor auth"))).kind == "miss"
    run(go())


def test_bypass_rules():
    async def go():
        c, _ = make()
        assert (await c.lookup(req("what is the weather today"))).reason == "time-sensitive"
        assert (await c.lookup(req("hi", tools=[{}]))).reason == "tools"
        assert (await c.lookup(req("hi", temperature=1.8))).reason == "temperature"
    run(go())


def test_ttl_expiry():
    async def go():
        c, _ = make()
        await c.put(req("explain tcp handshake"), RESP, ttl=1)
        assert (await c.lookup(req("explain tcp handshake"))).hit
        await asyncio.sleep(1.1)
        assert not (await c.lookup(req("explain tcp handshake"))).hit
    run(go())


def test_single_flight():
    async def go():
        c, _ = make()
        calls = 0

        async def f():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.05)
            return {"ok": 1}
        res = await asyncio.gather(*[c.single_flight("k", f) for _ in range(5)])
        assert calls == 1 and sum(1 for _, lead in res if lead) == 1
    run(go())


def test_api_end_to_end():
    with TestClient(app) as cl:
        body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "How do I reset my password?"}]}
        r1 = cl.post("/v1/chat/completions", json=body)
        assert r1.headers["x-cache"] == "MISS"
        t = time.perf_counter()
        r2 = cl.post("/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "how can I reset my password"}]})
        assert r2.headers["x-cache"] == "HIT-SEMANTIC" and r2.json()["choices"] == r1.json()["choices"]
        r3 = cl.post("/v1/chat/completions", json={**body, "stream": True})
        assert r3.headers["x-cache"].startswith("HIT") and "[DONE]" in r3.text
        assert cl.get("/admin/stats").json()["hits"] == 2
        assert "semcache_requests_total" in cl.get("/metrics").text
