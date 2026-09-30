"""Prometheus metrics, in-process stats for the dashboard, and cost accounting."""
import time
from collections import deque

from prometheus_client import Counter, Gauge, Histogram

REQS = Counter("semcache_requests_total", "Proxy requests", ["result", "model"])
LAT = Histogram("semcache_request_seconds", "End-to-end latency", ["result"],
                buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2, 5, 10, 30))
SIM = Histogram("semcache_similarity", "Best similarity score seen at lookup",
                buckets=(.5, .6, .7, .75, .8, .85, .9, .92, .95, .98, 1.0))
TOK = Counter("semcache_tokens_saved_total", "Tokens not re-generated thanks to cache hits", ["model"])
USD = Counter("semcache_usd_saved_total", "Estimated USD saved by cache hits")
ENTRIES = Gauge("semcache_entries", "Entries in the vector store")
COALESCED = Counter("semcache_coalesced_total", "Requests served by joining an in-flight upstream call")

# USD per 1M tokens (input, output). Approximate; override for your own contracts.
PRICES = {"gpt-4o": (2.5, 10.0), "gpt-4o-mini": (0.15, 0.6), "gpt-4.1": (2.0, 8.0), "gpt-4.1-mini": (0.4, 1.6)}


def cost_usd(model: str, usage: dict) -> float:
    pin, pout = next((v for k, v in PRICES.items() if (model or "").startswith(k)), (1.0, 3.0))
    return ((usage.get("prompt_tokens", 0) * pin) + (usage.get("completion_tokens", 0) * pout)) / 1e6


class Stats:
    def __init__(self):
        self.n = {"exact": 0, "semantic": 0, "miss": 0, "bypass": 0, "coalesced": 0}
        self.tokens_saved, self.usd_saved = 0, 0.0
        self.ms = {"hit": [0, 0.0], "miss": [0, 0.0]}
        self.events: deque = deque(maxlen=40)

    def record(self, kind, model, ms, sim, usage, prompt, coalesced=False):
        self.n[kind] += 1
        REQS.labels(kind if kind in ("miss", "bypass") else f"hit_{kind}", model or "?").inc()
        LAT.labels(kind if kind in ("miss", "bypass") else f"hit_{kind}").observe(ms / 1000)
        if sim:
            SIM.observe(sim)
        bucket = "hit" if kind in ("exact", "semantic") else "miss"
        self.ms[bucket][0] += 1
        self.ms[bucket][1] += ms
        if coalesced:
            self.n["coalesced"] += 1
            COALESCED.inc()
        if bucket == "hit":
            toks = usage.get("total_tokens") or (usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
            usd = cost_usd(model, usage)
            self.tokens_saved += toks
            self.usd_saved += usd
            TOK.labels(model or "?").inc(toks)
            USD.inc(usd)
        self.events.appendleft({"ts": time.time(), "kind": "coalesced" if coalesced else kind, "sim": round(sim, 4),
                                "ms": round(ms, 1), "model": model, "prompt": (prompt or "")[:90]})

    def snapshot(self, entries, cfg, emb_name, store_name):
        hits = self.n["exact"] + self.n["semantic"]
        served = hits + self.n["miss"]
        avg = lambda b: round(self.ms[b][1] / self.ms[b][0], 1) if self.ms[b][0] else 0
        return {**self.n, "hits": hits, "hit_rate": round(hits / served, 4) if served else 0,
                "tokens_saved": self.tokens_saved, "usd_saved": round(self.usd_saved, 6),
                "avg_hit_ms": avg("hit"), "avg_miss_ms": avg("miss"), "entries": entries,
                "events": list(self.events),
                "config": {"store": store_name, "embedder": emb_name, "threshold": cfg.threshold, "ttl": cfg.ttl}}
