"""
Data loader that streams tokenized data from parquet files and packs into (micro)batches to be used in training.

The goal is to pack many independent document segments into one fixed-shape row.
Each segment has a maximum length to keep the FlexAttention block mask efficient
and prevent individual documents from dominating the batch.

We do not split a segment just to fill the remaining space in a microbatch.
Instead, each packed row is padded to tokens_per_batch with labels=-100 and
segment_ids=-1. This keeps shapes fixed while preserving hard document boundaries.
"""

import os
import random

import pyarrow.parquet as pq
import torch

from .common import get_base_dir


def parquet_doc_segments(parquet_path, token_col="input_ids", T=512, seed=None):
    """
    Yields token segments (as plain python lists), each chunk is from ONE document only.
    Chunk length is in [2, T+1].
    Row groups are read in random order to keep streaming simple while avoiding
    deterministic ordering across files. 
    This shuffling is necessary since the way we sample climbmix leads to our raw data
    being clustered. Choosing a low enough row group size is also important to ensure
    a diversity of clusters in each batch.
    """
    pf = pq.ParquetFile(parquet_path)

    if pf.num_row_groups == 0:
        return

    max_chunk_len = T + 1
    
    rng = random.Random(seed)
    row_group_perm = rng.sample(range(pf.num_row_groups), pf.num_row_groups)

    for rg_idx in row_group_perm:
        rb = pf.read_row_group(rg_idx, columns=[token_col])
        col = rb.column(0)

        for row in col:
            toks = row.as_py()
            if not toks or len(toks) < 2: # Skip empty rows or rows with less than 2 tokens. This should never happen.
                continue

            for start in range(0, len(toks) - 1, T):
                chunk = toks[start : start + max_chunk_len]
                if len(chunk) >= 2:
                    yield chunk


def count_dataset_tokens(parquet_path, token_col="input_ids"):
    """Count total training tokens (next-token targets) in a parquet dataset."""
    pf = pq.ParquetFile(parquet_path)
    total_tokens = 0
    for rg_idx in range(pf.num_row_groups):
        rb = pf.read_row_group(rg_idx, columns=[token_col])
        col = rb.column(0)
        for row in col:
            toks = row.as_py()
            if toks and len(toks) >= 2:
                total_tokens += len(toks) - 1
    return total_tokens


def pack_batch(segments, tokens_per_batch, pad_token_id=0, device=None):
    """Pack Python-list segments into a FlexAttention batch.

    segments: list[list[int]], each segment length >= 2 and <= T

    Returns:
      input_ids    [1, tokens_per_batch]  (long)
      labels       [1, tokens_per_batch]  (long, pad positions are -100)
      segment_ids  [1, tokens_per_batch]  (long, independent document segments; pad is -1)
      position_ids [1, tokens_per_batch]  (long, resets each segment)

    """

    input_ids: list[int] = []
    labels: list[int] = []
    segment_ids: list[int] = []
    position_ids: list[int] = []

    for segment_id, s in enumerate(segments):
        L = len(s) - 1  # after shift

        # x = s[:-1], y = s[1:]
        input_ids.extend(s[:-1])
        labels.extend(s[1:])
        segment_ids.extend([segment_id] * L)
        position_ids.extend(range(L))

    pad_len = tokens_per_batch - len(input_ids)
    assert pad_len >= 0 # Caller should ensure segments fit in tokens_per_batch, impossible for batch_iterator as caller.
    input_ids.extend([pad_token_id] * pad_len)
    labels.extend([-100] * pad_len)
    segment_ids.extend([-1] * pad_len)
    position_ids.extend([0] * pad_len)

    # CUDA and ROCm supports memory pinning for asynchronous transfers between CPU and GPU
    # I have absolutely no idea if doing it this way is any faster than creating the tensor on GPU directly
    # It probably does not matter much, and i spent way too much time on it, so i am leaving it as it is
    # PyTorch reports AMD ROCm accelerators as cuda devices too.
    use_pinned_memory = device is not None and torch.device(device).type == "cuda"
    input_ids_t = torch.tensor(input_ids, dtype=torch.long, pin_memory=use_pinned_memory, device="cpu")
    labels_t = torch.tensor(labels, dtype=torch.long, pin_memory=use_pinned_memory, device="cpu")
    segment_ids_t = torch.tensor(segment_ids, dtype=torch.long, pin_memory=use_pinned_memory, device="cpu")
    position_ids_t = torch.tensor(position_ids, dtype=torch.long, pin_memory=use_pinned_memory, device="cpu")

    if device is not None:
        input_ids_t = input_ids_t.to(device, non_blocking=use_pinned_memory)
        labels_t = labels_t.to(device, non_blocking=use_pinned_memory)
        segment_ids_t = segment_ids_t.to(device, non_blocking=use_pinned_memory)
        position_ids_t = position_ids_t.to(device, non_blocking=use_pinned_memory)

    # FlexAttention expects batched tensors; each loader batch is one packed row.
    return input_ids_t.unsqueeze(0), labels_t.unsqueeze(0), segment_ids_t.unsqueeze(0), position_ids_t.unsqueeze(0)


def batch_iterator(
    parquet_path,
    *,
    tokens_per_batch: int,
    max_sl: int = 512,
    token_col: str = "input_ids",
    pad_token_id: int = 0,
    drop_last: bool = True,
    device="cuda",
    seed=None,
    rank: int = 0,
    world_size: int = 1,
):
    """Yield packed (micro)batches with token budget `tokens_per_batch`.

    We stream doc-segments (each <= T+1 tokens raw, so <= T after shift) and pack
    them until the sum of training positions (len(chunk)-1) reaches
    tokens_per_batch.

    If the next segment would exceed the token budget, the current row is padded
    to tokens_per_batch and yielded. Segments are never split by batch packing.

    If world_size is more than 1, each rank yields its own strided subset of batches. 
    Simple and deterministic way to support multi-gpu.
    """

    assert max_sl <= tokens_per_batch

    buf: list[list[int]] = []
    tok = 0  # sum of (len(chunk)-1) in buf
    batch_idx = 0

    for chunk in parquet_doc_segments(parquet_path, token_col=token_col, T=max_sl, seed=seed):
        seglen = len(chunk) - 1
        if seglen <= 0:
            continue

        if buf and tok + seglen > tokens_per_batch:
            if batch_idx % world_size == rank:
                yield pack_batch(buf, tokens_per_batch=tokens_per_batch, pad_token_id=pad_token_id, device=device)
            batch_idx += 1
            buf.clear()
            tok = 0

        buf.append(chunk)
        tok += seglen

    if buf and not drop_last:
        if batch_idx % world_size == rank:
            yield pack_batch(buf, tokens_per_batch=tokens_per_batch, pad_token_id=pad_token_id, device=device)


# DEBUG
if __name__ == "__main__":
    parquet_file = os.path.join(get_base_dir(), "data", "tokenized", "climbmix100Mwords.parquet")
    batch_count = 5
    i = 0
    for input_ids, labels, segment_ids, position_ids in batch_iterator(
        parquet_file,
        tokens_per_batch=8192,
        max_sl=256,
        token_col="input_ids",
        drop_last=True
    ):
        i += 1
        print(input_ids.shape, labels.shape, segment_ids.shape, position_ids.shape)
        if i > batch_count:
            break
