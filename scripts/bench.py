"""Load-test the proxy with paraphrased prompts and report hit rate / latency / savings.
Usage: python scripts/bench.py --url http://localhost:8000 --n 200"""
import argparse
import random
import statistics as S
import time

import httpx

FAMILIES = [
    ["How do I reset my password?", "how can I reset my password", "Steps to reset password please", "I forgot my password, how to reset it"],
    ["Explain the TCP three-way handshake", "how does the tcp 3 way handshake work", "What is the TCP handshake process"],
    ["What is a vector database?", "explain vector databases", "Tell me what a vector database is"],
    ["Difference between REST and GraphQL", "REST vs GraphQL differences", "how is graphql different from rest"],
    ["Write a python function to reverse a string", "python function that reverses a string", "reverse a string in python function"],
]

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--n", type=int, default=100)
a = ap.parse_args()
random.seed(7)
rows = []
with httpx.Client(timeout=120) as c:
    for _ in range(a.n):
        q = random.choice(random.choice(FAMILIES))
        t = time.perf_counter()
        r = c.post(f"{a.url}/v1/chat/completions", json={"model": "gpt-4o-mini", "messages": [{"role": "user", "content": q}]})
        rows.append((r.headers.get("x-cache", "?"), (time.perf_counter() - t) * 1000))
hits = [ms for k, ms in rows if k.startswith("HIT")]
miss = [ms for k, ms in rows if not k.startswith("HIT")]
print(f"requests={len(rows)}  hit-rate={len(hits)/len(rows):.1%}")
if hits:
    print(f"hit  p50={S.median(hits):.1f}ms  p95={sorted(hits)[int(len(hits)*.95)-1]:.1f}ms")
if miss:
    print(f"miss p50={S.median(miss):.1f}ms")
if hits and miss:
    print(f"speed-up on hits ≈ {S.median(miss)/S.median(hits):.0f}x")
