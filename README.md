# LLMOps Nexus — Semantic Caching Layer for LLM APIs

> An LLMOps infrastructure project: semantic cache · observability · cost accounting.

Drop-in **OpenAI-compatible proxy** that serves repeated *and paraphrased* prompts from a vector cache, cutting LLM cost and latency.

**Live demo (GitHub Pages):** `https://nikhil-creat.github.io/llmops-nexus/` — the full cache algorithm running in your browser.

*Designed and Developed by*
# **NIKHIL CHARY SRIRAMOJU**
BTech CSE (Final Year)

- GitHub: [Nikhil-creat](https://github.com/Nikhil-creat)
- LinkedIn: [nikhil-chary-sriramoju](https://in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a)
- Email: sriramojunikhil66@gmail.com
- Instagram: [@nikhil__sriramoju](https://www.instagram.com/nikhil__sriramoju)
- Facebook: [Profile](https://www.facebook.com/profile.php?id=100079201124141)

## Features
- **Two-tier cache:** L1 exact-hash LRU → L2 vector similarity (Redis Stack HNSW, Qdrant, or in-memory)
- **Embeddings:** OpenAI `text-embedding-3-small`, or a deterministic local hashing embedder (offline / CI)
- **Safety guards:** scope isolation per model + system prompt + params · number & negation mismatch rejection · freshness guard for time-sensitive prompts · tools / multimodal / high-temperature bypass
- **Adaptive TTL** by temperature; per-request overrides via headers
- **Single-flight coalescing:** identical concurrent requests share one upstream call
- **Streaming:** real SSE pass-through on miss (cached afterwards), SSE replay on hit
- **Observability:** Prometheus metrics + provisioned Grafana dashboard + built-in live dashboard (`/dashboard`)
- **Cost accounting:** tokens and USD saved per hit

## Quick start
```bash
cp .env.example .env            # MOCK_UPSTREAM=1 works without any API key
docker compose up --build       # API :8000 · Grafana :3000 · Prometheus :9090
python scripts/bench.py --n 100 # hit-rate / latency report
```
Without Docker: `pip install -r requirements.txt && VECTOR_STORE=memory MOCK_UPSTREAM=1 uvicorn app.main:app --reload`

Point any OpenAI SDK at it: `base_url="http://localhost:8000/v1"`.

## Headers
| Direction | Header | Meaning |
|---|---|---|
| response | `X-Cache` | `HIT-EXACT` · `HIT-SEMANTIC` · `MISS` · `BYPASS` |
| response | `X-Cache-Similarity`, `X-Cache-Saved-Ms` | best score, latency saved |
| request | `Cache-Control: no-cache` | skip cache, refresh entry |
| request | `X-Cache-Threshold`, `X-Cache-TTL` | per-request overrides |

## Endpoints
`POST /v1/chat/completions` · `GET /dashboard` · `GET /metrics` · `GET /admin/stats` · `DELETE /admin/cache` · `GET /healthz` · `GET /about`

## Tuning
Start at threshold **0.92** with OpenAI embeddings (0.80 for the local embedder). Watch `semcache_similarity` in Grafana: if near-misses cluster just below the threshold and answers are interchangeable, lower it; if users see wrong answers, raise it.

## Notes
- Qdrant has no native TTL; expiry is enforced at read time.
- Prices in `app/metrics.py` are approximate — edit for your contract.
- GitHub Pages hosts only the static simulator (`/docs`); the FastAPI backend runs via Docker / any container host.

MIT License
