"""LLMOps Nexus — drop-in OpenAI-compatible proxy with a semantic cache in front."""
import asyncio
import json
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import metrics as M
from .cache import SemanticCache
from .config import Settings
from .embedder import build_embedder
from .stores import build_store
from .upstream import UpstreamError, assemble, build_upstream

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Settings()
    client = httpx.AsyncClient(timeout=120)
    emb = build_embedder(cfg, client)
    store = build_store(cfg, emb.dim)
    await store.init()
    app.state.cfg, app.state.emb, app.state.client = cfg, emb, client
    app.state.cache = SemanticCache(cfg, emb, store)
    app.state.up = build_upstream(cfg, client)
    app.state.stats = M.Stats()

    async def gauge():
        while True:
            try:
                M.ENTRIES.set(await store.count())
            except Exception:
                pass
            await asyncio.sleep(5)

    task = asyncio.create_task(gauge())
    yield
    task.cancel()
    await client.aclose()


app = FastAPI(title="LLMOps Nexus", version="1.0.0", lifespan=lifespan)


def sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


def admin(req: Request):
    tok = req.app.state.cfg.admin_token
    if tok and req.headers.get("x-admin-token") != tok:
        raise HTTPException(401, "invalid admin token")


async def replay(resp: dict):
    """Re-emit a cached completion as an SSE stream so streaming clients work on hits."""
    msg = resp["choices"][0]["message"].get("content") or ""
    base = {"id": resp["id"], "object": "chat.completion.chunk", "created": resp["created"], "model": resp["model"]}
    yield sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    parts = msg.split(" ")
    for i, w in enumerate(parts):
        yield sse({**base, "choices": [{"index": 0, "delta": {"content": w + (" " if i < len(parts) - 1 else "")}, "finish_reason": None}]})
    yield sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": resp["choices"][0].get("finish_reason", "stop")}]})
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat(req: Request):
    st = req.app.state
    cache, up, stats = st.cache, st.up, st.stats
    body = await req.json()
    t0, auth = time.perf_counter(), req.headers.get("authorization")
    stream, model = bool(body.get("stream")), body.get("model")
    thr = float(req.headers["x-cache-threshold"]) if "x-cache-threshold" in req.headers else None
    ttl = int(req.headers["x-cache-ttl"]) if "x-cache-ttl" in req.headers else None
    refresh = "no-cache" in req.headers.get("cache-control", "").lower()
    prompt = cache.query_text(body)

    look = await cache.lookup(body, thr)
    if refresh and look.hit:
        look.kind, look.payload = "bypass", None
        look.reason = "client no-cache"

    if look.hit:
        resp = look.payload["response"]
        ms = (time.perf_counter() - t0) * 1000
        stats.record(look.kind, model, ms, look.similarity, resp.get("usage", {}), prompt)
        h = {"X-Cache": f"HIT-{look.kind.upper()}", "X-Cache-Similarity": f"{look.similarity:.4f}",
             "X-Cache-Saved-Ms": f"{max(0.0, look.payload.get('latency_ms', 0) - ms):.0f}"}
        if stream:
            return StreamingResponse(replay(resp), media_type="text/event-stream", headers=h)
        return JSONResponse({**resp, "x_semantic_cache": {"kind": look.kind, "similarity": round(look.similarity, 4)}}, headers=h)

    kind = "bypass" if look.kind == "bypass" else "miss"
    h = {"X-Cache": kind.upper(), "X-Cache-Similarity": f"{look.similarity:.4f}"}
    if look.reason:
        h["X-Cache-Bypass-Reason"] = look.reason

    if stream:
        async def gen():
            chunks, t1 = [], time.perf_counter()
            want_usage = (body.get("stream_options") or {}).get("include_usage")
            try:
                async for c in up.stream(body, auth):
                    chunks.append(c)
                    if c.get("choices") or want_usage:
                        yield sse(c)
            except UpstreamError as e:
                yield sse({"error": {"message": e.detail, "status": e.status}})
                return
            yield "data: [DONE]\n\n"
            resp = assemble(chunks, body)
            ms = (time.perf_counter() - t0) * 1000
            if look.cacheable:
                await cache.put(body, resp, ttl, (time.perf_counter() - t1) * 1000)
            stats.record(kind, model, ms, look.similarity, resp["usage"], prompt)

        return StreamingResponse(gen(), media_type="text/event-stream", headers=h)

    async def fill():
        t1 = time.perf_counter()
        r = await up.chat(body, auth)
        if look.cacheable:
            await cache.put(body, r, ttl, (time.perf_counter() - t1) * 1000)
        return r

    try:
        resp, leader = await cache.single_flight(cache.exact_key(body), fill)
    except UpstreamError as e:
        return Response(e.detail, status_code=e.status, media_type="application/json")
    ms = (time.perf_counter() - t0) * 1000
    stats.record(kind, model, ms, look.similarity, resp.get("usage", {}), prompt, coalesced=not leader)
    return JSONResponse(resp, headers=h)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/admin/stats")
async def admin_stats(req: Request):
    admin(req)
    st = req.app.state
    return st.stats.snapshot(await st.cache.store.count(), st.cfg, st.emb.name, st.cache.store.name)


@app.delete("/admin/cache")
async def admin_flush(req: Request):
    admin(req)
    await req.app.state.cache.flush()
    return {"flushed": True}


@app.get("/dashboard")
async def dashboard():
    return FileResponse(STATIC / "dashboard.html")


@app.get("/")
async def root():
    return RedirectResponse("/dashboard")


@app.get("/about")
async def about():
    return {"project": "LLMOps Nexus — Semantic Caching Layer for LLM APIs", "version": app.version,
            "designed_and_developed_by": "NIKHIL CHARY SRIRAMOJU", "education": "BTech CSE (Final Year)",
            "github": "https://github.com/Nikhil-creat", "linkedin": "https://in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a",
            "email": "sriramojunikhil66@gmail.com", "instagram": "https://www.instagram.com/nikhil__sriramoju",
            "facebook": "https://www.facebook.com/profile.php?id=100079201124141"}
