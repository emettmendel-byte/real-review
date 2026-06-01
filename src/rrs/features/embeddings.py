"""MiniLM embeddings + per-user / per-business max-similarity scalars.

`all-MiniLM-L6-v2` produces 384-dim L2-normalized vectors. We persist them to
`features/embeddings.npy` (float32 memmap) plus an `embeddings_index.parquet` mapping
review_id → row index, so Phase 4 modeling can decide whether to consume them raw,
PCA-reduce them, or only use the two derived similarity scalars we compute here:

- `max_sim_to_user_history`     max cosine to another review by the same user
- `max_sim_to_business_reviews` max cosine to another review on the same business

Both are 0 when there's nothing to compare against (one-shot user, single-review biz).

GPU: on Apple Silicon the model runs on `mps`, taking the 1.9M-review encode from
several hours on CPU to under an hour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

MODEL_NAME = "all-MiniLM-L6-v2"
EMB_DIM = 384


def _select_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def encode_texts(
    texts: list[str],
    *,
    batch_size: int = 128,
    out_path: Path | None = None,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed `texts` with MiniLM, return float32 array shape (N, 384), L2-normalized.

    If `out_path` is given, also writes the array to disk as a .npy memmap-able file.
    """
    from sentence_transformers import SentenceTransformer

    device = _select_device()
    print(f"  encoding {len(texts):,} texts on device={device}", flush=True)
    model = SentenceTransformer(MODEL_NAME, device=device)
    embs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32, copy=False)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, embs, allow_pickle=False)
        print(f"  wrote {out_path} ({embs.nbytes / 1e9:.2f} GB)", flush=True)
    return embs


def max_sim_within_groups(
    keys: np.ndarray,
    embs: np.ndarray,
    *,
    label: str = "group",
) -> np.ndarray:
    """For each row, max cosine similarity to any *other* row sharing the same key.

    Lone-key rows get 0. `embs` must be L2-normalized so dot-product == cosine.

    Implementation: argsort by key once to get contiguous groups, then for each group
    compute `X @ X.T`, zero the diagonal, take row-wise max. Memory for the biggest
    group dominates — for Philadelphia the largest user-group is ~3K reviews
    (3000² × 4 B ≈ 36 MB) and the largest biz-group ~5.8K (5800² × 4 B ≈ 134 MB).
    """
    n = len(keys)
    if n != len(embs):
        raise ValueError("keys and embs must align")
    out = np.zeros(n, dtype=np.float32)

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    # Boundaries between groups in the sorted array.
    edges = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1], True])

    n_groups = len(edges) - 1
    log_every = max(1, n_groups // 20)
    biggest = 0
    for gi in range(n_groups):
        start, stop = edges[gi], edges[gi + 1]
        if stop - start < 2:
            continue
        idx = order[start:stop]
        X = embs[idx]  # (m, 384)
        sim = X @ X.T  # (m, m)
        # Self-similarity is 1.0 — exclude it by setting the diagonal to -inf.
        np.fill_diagonal(sim, -np.inf)
        out[idx] = sim.max(axis=1).astype(np.float32, copy=False)
        biggest = max(biggest, stop - start)
        if gi % log_every == 0:
            print(
                f"    {label}: {gi:,}/{n_groups:,} groups (biggest so far: {biggest})",
                flush=True,
            )
    print(f"    {label}: done, biggest group = {biggest}", flush=True)
    return out


def compute_similarities(
    embs: np.ndarray,
    user_ids: list[str],
    business_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute both per-user and per-business max-sim scalars."""
    if not (len(embs) == len(user_ids) == len(business_ids)):
        raise ValueError("embs, user_ids, business_ids must align")
    users = np.asarray(user_ids)
    bizes = np.asarray(business_ids)
    print("  max similarity within each user's history ...", flush=True)
    max_user = max_sim_within_groups(users, embs, label="user-sim")
    print("  max similarity within each business's reviews ...", flush=True)
    max_biz = max_sim_within_groups(bizes, embs, label="biz-sim")
    return max_user, max_biz


def write_embedding_index(review_ids: list[str], out_dir: Path) -> None:
    """Write an `embeddings_index.parquet` mapping review_id → row index of the .npy."""
    df = pl.DataFrame({"review_id": review_ids, "row": pl.arange(0, len(review_ids), eager=True)})
    out_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_dir / "embeddings_index.parquet")


def estimate_runtime_hours(n_texts: int, throughput_per_sec: float = 800.0) -> float:
    """Rough wall-clock estimate for the encode step. Default throughput is the
    observed rate on Apple M-series MPS for short-to-medium reviews."""
    return n_texts / throughput_per_sec / 3600.0


__all__ = [
    "EMB_DIM",
    "MODEL_NAME",
    "compute_similarities",
    "encode_texts",
    "estimate_runtime_hours",
    "max_sim_within_groups",
    "write_embedding_index",
]
