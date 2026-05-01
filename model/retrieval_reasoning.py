import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss
except Exception:  # pragma: no cover - handled at runtime
    faiss = None


@dataclass
class RetrievalOutput:
    indices: torch.Tensor
    scores: torch.Tensor
    attention_weights: Optional[torch.Tensor]


class QueryProjector(nn.Module):
    def __init__(self, dim: int, query_type: str = "sum"):
        super().__init__()
        self.query_type = query_type
        if query_type == "mlp":
            self.proj = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.ReLU(),
                nn.Linear(dim, dim),
            )
        elif query_type == "concat_linear":
            self.proj = nn.Linear(dim * 2, dim)
        else:
            self.proj = None

    def forward(self, drug_emb: torch.Tensor, disease_emb: torch.Tensor) -> torch.Tensor:
        if self.query_type == "sum":
            return drug_emb + disease_emb
        concat = torch.cat([drug_emb, disease_emb], dim=-1)
        return self.proj(concat)


class AttentionReasoner(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, query: torch.Tensor, neighbors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(query).unsqueeze(1)
        k = self.k_proj(neighbors)
        scores = (q * k).sum(-1) * self.scale
        weights = F.softmax(scores, dim=-1)
        context = torch.sum(weights.unsqueeze(-1) * neighbors, dim=1)
        return context, weights


class FaissRetriever:
    def __init__(self, dim: int, use_gpu: bool = True):
        self.dim = dim
        self.use_gpu = use_gpu
        self.index = None
        self.gpu_resources = None
        self.embedding_bank = None

    def _build_index(self) -> None:
        index = faiss.IndexFlatIP(self.dim)
        if self.use_gpu and faiss is not None and faiss.get_num_gpus() > 0:
            self.gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
        self.index = index

    def build(self, embeddings: torch.Tensor) -> None:
        if faiss is None:
            raise RuntimeError("FAISS is not available. Install faiss-gpu or faiss-cpu.")
        if self.index is None:
            self._build_index()
        self.index.reset()
        emb = embeddings.detach().float()
        emb = F.normalize(emb, dim=-1)
        self.embedding_bank = emb
        emb_np = emb.detach().cpu().numpy().astype("float32")
        self.index.add(emb_np)

    def search(self, query: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.index is None:
            raise RuntimeError("FAISS index is not built.")
        q = F.normalize(query.detach().float(), dim=-1)
        q_np = q.detach().cpu().numpy().astype("float32")
        scores_np, indices_np = self.index.search(q_np, top_k)
        indices = torch.from_numpy(indices_np).to(query.device)
        scores = torch.from_numpy(scores_np).to(query.device)
        return indices, scores


class RetrievalReasoner(nn.Module):
    def __init__(
        self,
        dim: int,
        top_k: int,
        mode: str,
        query_type: str,
        use_gpu: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.mode = mode
        self.query_projector = QueryProjector(dim, query_type=query_type)
        self.attention = AttentionReasoner(dim)
        self.retriever = FaissRetriever(dim, use_gpu=use_gpu)

    def build_index(self, embedding_bank: torch.Tensor) -> None:
        self.retriever.build(embedding_bank)

    def forward(
        self,
        drug_emb: torch.Tensor,
        disease_emb: torch.Tensor,
        embedding_bank: torch.Tensor,
    ) -> Tuple[torch.Tensor, RetrievalOutput]:
        query = self.query_projector(drug_emb, disease_emb)
        top_k = min(self.top_k, embedding_bank.shape[0])
        indices, scores = self.retriever.search(query, top_k)
        neighbors = embedding_bank[indices]

        if self.mode == "retrieval":
            context = neighbors.mean(dim=1)
            weights = torch.full_like(scores, 1.0 / top_k)
        else:
            context, weights = self.attention(query, neighbors)

        return context, RetrievalOutput(indices=indices, scores=scores, attention_weights=weights)
