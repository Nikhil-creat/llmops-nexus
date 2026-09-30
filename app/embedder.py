"""Embedding providers: OpenAI text-embedding-3-small, or a deterministic local
hashing embedder so the whole system runs offline (demos, CI, tests)."""
import hashlib
import re

import httpx
import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the to of is are was be do does did how what why can could would should i me my you your we "
    "please tell explain for in on with and or it this that there any some about give show".split()
)


def _h(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


def _stem(t: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 3 and t.endswith(suf):
            return t[: -len(suf)]
    return t


class LocalEmbedder:
    dim = 384
    name = "local-hash-384"

    async def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, np.float32)
        toks = [_stem(t) for t in _TOKEN.findall(text.lower()) if t not in _STOP]
        for t in toks:
            v[_h("w:" + t) % self.dim] += 2.0
            g = f"^{t}$"
            for i in range(len(g) - 2):
                v[_h("c:" + g[i : i + 3]) % self.dim] += 0.5
        for a, b in zip(toks, toks[1:]):
            v[_h("b:" + a + b) % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v


class OpenAIEmbedder:
    def __init__(self, client: httpx.AsyncClient, base: str, key: str, model: str, dim: int):
        self.client, self.base, self.key, self.model, self.dim = client, base, key, model, dim
        self.name = model

    async def embed(self, text: str) -> np.ndarray:
        r = await self.client.post(
            f"{self.base}/embeddings",
            headers={"Authorization": f"Bearer {self.key}"},
            json={"model": self.model, "input": text},
        )
        r.raise_for_status()
        v = np.asarray(r.json()["data"][0]["embedding"], np.float32)
        return v / (np.linalg.norm(v) or 1.0)


def build_embedder(cfg, client):
    if cfg.embedder == "openai":
        return OpenAIEmbedder(client, cfg.upstream_base, cfg.upstream_key, cfg.embed_model, cfg.embed_dim)
    return LocalEmbedder()
