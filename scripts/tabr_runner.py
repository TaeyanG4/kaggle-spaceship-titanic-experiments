"""TabR on GPU without faiss-gpu (H-A-29).

pytabkit's TabR asks faiss for a GPU flat-L2 index on CUDA, but only faiss-cpu is installed on
Windows. This shim supplies an exact torch replacement (squared L2 distances + topk on the GPU),
which returns the same neighbours as `faiss.GpuIndexFlatL2`, then runs `member_runner` model `tabr`.
"""

from __future__ import annotations


class _GpuIndexFlatConfig:
    device = 0


class _TorchFlatL2:
    def __init__(self, resources, dim, config=None):
        self.keys = None

    def reset(self):
        self.keys = None

    def add(self, keys):
        self.keys = keys

    def search(self, queries, k):
        import torch

        distances = torch.cdist(queries.float(), self.keys.float()).pow(2)
        values, indices = distances.topk(k, dim=1, largest=False)
        return values, indices


def run_tabr(name: str, seed: int, threads: int) -> dict:
    import faiss
    from member_runner import run_member

    faiss.GpuIndexFlatConfig = _GpuIndexFlatConfig
    faiss.GpuIndexFlatL2 = _TorchFlatL2
    faiss.StandardGpuResources = lambda: None
    return run_member(name, seed, "tabr", threads)
