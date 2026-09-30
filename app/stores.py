"""Vector store backends behind one interface: memory | redis (RediSearch HNSW) | qdrant."""
import json
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class Hit:
    score: float  # cosine similarity, 1.0 = identical
    payload: dict


class MemoryStore:
    name = "memory"

    def __init__(self, dim: int, max_items: int = 50_000):
        self.max = max_items
        self.items: dict[str, tuple[np.ndarray, str, dict, float]] = {}

    async def init(self): ...

    async def search(self, vec, scope):
        now, best = time.time(), None
        for k, (v, sc, p, exp) in list(self.items.items()):
            if exp < now:
                del self.items[k]
                continue
            if sc != scope:
                continue
            s = float(v @ vec)
            if best is None or s > best.score:
                best = Hit(s, p)
        return best

    async def upsert(self, id, vec, scope, payload, ttl):
        if len(self.items) >= self.max:
            self.items.pop(next(iter(self.items)))
        self.items[id] = (vec, scope, payload, time.time() + ttl)

    async def count(self):
        return len(self.items)

    async def flush(self):
        self.items.clear()


class RedisStore:
    """Redis Stack: HNSW vector index + TAG filter for scope, native key TTL."""

    name = "redis"

    def __init__(self, url: str, dim: int, index: str = "semcache"):
        import redis.asyncio as redis

        self.r, self.dim, self.index = redis.from_url(url), dim, index

    async def init(self):
        try:
            await self.r.execute_command("FT.INFO", self.index)
        except Exception:
            await self.r.execute_command(
                "FT.CREATE", self.index, "ON", "HASH", "PREFIX", "1", "sc:", "SCHEMA",
                "scope", "TAG", "vec", "VECTOR", "HNSW", "6",
                "TYPE", "FLOAT32", "DIM", str(self.dim), "DISTANCE_METRIC", "COSINE",
            )

    async def search(self, vec, scope):
        q = f"(@scope:{{{scope}}})=>[KNN 1 @vec $v AS dist]"
        res = await self.r.execute_command(
            "FT.SEARCH", self.index, q, "PARAMS", "2", "v", vec.astype(np.float32).tobytes(),
            "SORTBY", "dist", "RETURN", "2", "dist", "payload", "DIALECT", "2",
        )
        if not res or res[0] == 0:
            return None
        f = {(k.decode() if isinstance(k, bytes) else k): v for k, v in zip(res[2][::2], res[2][1::2])}
        return Hit(1.0 - float(f["dist"]), json.loads(f["payload"]))

    async def upsert(self, id, vec, scope, payload, ttl):
        key = f"sc:{id}"
        await self.r.hset(key, mapping={"scope": scope, "vec": vec.astype(np.float32).tobytes(), "payload": json.dumps(payload)})
        await self.r.expire(key, int(ttl))

    async def count(self):
        info = await self.r.execute_command("FT.INFO", self.index)
        d = dict(zip(info[::2], info[1::2]))
        return int(d.get(b"num_docs", d.get("num_docs", 0)))

    async def flush(self):
        await self.r.execute_command("FT.DROPINDEX", self.index, "DD")
        await self.init()


class QdrantStore:
    """Qdrant: cosine HNSW with payload filter. TTL is enforced via expires_at at read time."""

    name = "qdrant"

    def __init__(self, url: str, dim: int, collection: str = "semcache"):
        from qdrant_client import AsyncQdrantClient

        self.c, self.dim, self.col = AsyncQdrantClient(url=url), dim, collection

    async def init(self):
        from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

        if not await self.c.collection_exists(self.col):
            await self.c.create_collection(self.col, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE))
            await self.c.create_payload_index(self.col, "scope", PayloadSchemaType.KEYWORD)

    async def search(self, vec, scope):
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        r = await self.c.query_points(
            self.col, query=vec.tolist(), limit=1, with_payload=True,
            query_filter=Filter(must=[FieldCondition(key="scope", match=MatchValue(value=scope))]),
        )
        return Hit(r.points[0].score, r.points[0].payload["data"]) if r.points else None

    async def upsert(self, id, vec, scope, payload, ttl):
        from qdrant_client.models import PointStruct

        await self.c.upsert(self.col, points=[PointStruct(id=id, vector=vec.tolist(), payload={"scope": scope, "data": payload})])

    async def count(self):
        return (await self.c.count(self.col)).count

    async def flush(self):
        await self.c.delete_collection(self.col)
        await self.init()


def build_store(cfg, dim):
    if cfg.store == "redis":
        return RedisStore(cfg.redis_url, dim)
    if cfg.store == "qdrant":
        return QdrantStore(cfg.qdrant_url, dim)
    return MemoryStore(dim)
