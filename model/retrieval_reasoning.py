from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import faiss
except Exception: 
    faiss = None


@dataclass
class EntityRetrievalOutput:
    indices: torch.Tensor
    scores: torch.Tensor
    attention_weights: Optional[torch.Tensor]


@dataclass
class PairRetrievalOutput:
    drug: EntityRetrievalOutput
    disease: EntityRetrievalOutput


class MultiHeadReasoner(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

    def forward(self, query: torch.Tensor, neighbors: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        context, weights = self.attention(
            query=query.unsqueeze(1),
            key=neighbors,
            value=neighbors,
            need_weights=True,
            average_attn_weights=True,
        )
        return context.squeeze(1), weights.squeeze(1)


class GatedFusion(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, original: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(torch.cat([original, context], dim=-1)))
        return gate * original + (1.0 - gate) * context


class FaissRetriever:
    def __init__(self, dim: int, use_gpu: bool = True):
        self.dim = dim
        self.use_gpu = use_gpu
        self.index = None
        self.gpu_resources = None
        self.torch_embeddings = None

    def _build_index(self) -> None:
        index = faiss.IndexFlatIP(self.dim)
        if self.use_gpu and faiss is not None and faiss.get_num_gpus() > 0:
            self.gpu_resources = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(self.gpu_resources, 0, index)
        self.index = index

    def build(self, embeddings: torch.Tensor) -> None:
        if faiss is None:
            with torch.no_grad():
                self.torch_embeddings = F.normalize(embeddings.detach().float(), dim=-1)
            return
        if self.index is None:
            self._build_index()
        self.index.reset()
        with torch.no_grad():
            emb = F.normalize(embeddings.detach().float(), dim=-1)
            emb_np = emb.cpu().numpy().astype("float32")
        self.index.add(emb_np)
        self.torch_embeddings = None

    def search(
        self,
        query: torch.Tensor,
        top_k: int,
        query_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.index is None and self.torch_embeddings is None:
            raise RuntimeError("FAISS index is not built.")

        search_k = top_k + 1 if query_ids is not None else top_k
        with torch.no_grad():
            q = F.normalize(query.detach().float(), dim=-1)
            if self.index is None:
                similarities = torch.matmul(q, self.torch_embeddings.to(query.device).t())
                scores, indices = torch.topk(similarities, k=search_k, dim=-1)
            else:
                q_np = q.cpu().numpy().astype("float32")
                scores_np, indices_np = self.index.search(q_np, search_k)
                indices = torch.from_numpy(indices_np).to(query.device)
                scores = torch.from_numpy(scores_np).to(query.device)

        if query_ids is None:
            return indices[:, :top_k], scores[:, :top_k]

        return self._remove_self_matches(indices, scores, query_ids, top_k)

    def _remove_self_matches(
        self,
        indices: torch.Tensor,
        scores: torch.Tensor,
        query_ids: torch.Tensor,
        top_k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        filtered_indices = []
        filtered_scores = []
        fallback_index = indices[:, -1]
        fallback_score = scores[:, -1]

        for row in range(indices.shape[0]):
            keep = indices[row] != query_ids[row]
            row_indices = indices[row][keep][:top_k]
            row_scores = scores[row][keep][:top_k]

            if row_indices.numel() < top_k:
                pad = top_k - row_indices.numel()
                row_indices = torch.cat([row_indices, fallback_index[row].repeat(pad)])
                row_scores = torch.cat([row_scores, fallback_score[row].repeat(pad)])

            filtered_indices.append(row_indices)
            filtered_scores.append(row_scores)

        return torch.stack(filtered_indices, dim=0), torch.stack(filtered_scores, dim=0)


class EntityRetrievalReasoner(nn.Module):
    def __init__(
        self,
        dim: int,
        top_k: int,
        mode: str,
        num_heads: int = 4,
        retrieval_dropout: float = 0.0,
        use_gpu: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.top_k = top_k
        self.mode = mode
        self.retrieval_dropout = retrieval_dropout
        self.reasoner = MultiHeadReasoner(dim, num_heads=num_heads)
        self.fusion = GatedFusion(dim)
        self.retriever = FaissRetriever(dim, use_gpu=use_gpu)

    def build_index(self, embedding_bank: torch.Tensor) -> None:
        self.retriever.build(embedding_bank)

    def forward(
        self,
        query: torch.Tensor,
        query_ids: torch.Tensor,
        embedding_bank: torch.Tensor,
    ) -> Tuple[torch.Tensor, EntityRetrievalOutput]:
        top_k = min(self.top_k, max(1, embedding_bank.shape[0] - 1))
        indices, scores = self.retriever.search(query, top_k, query_ids=query_ids)
        neighbors = embedding_bank[indices]

        if self.mode == "retrieval":
            context = neighbors.mean(dim=1)
            weights = torch.full_like(scores, 1.0 / top_k)
        else:
            context, weights = self.reasoner(query, neighbors)

        if self.training and self.retrieval_dropout > 0:
            keep = torch.rand(query.shape[0], 1, device=query.device) >= self.retrieval_dropout
            context = context * keep.to(context.dtype)

        enriched = self.fusion(query, context)
        return enriched, EntityRetrievalOutput(indices=indices, scores=scores, attention_weights=weights)


class DualEntityRetrievalReasoner(nn.Module):
    def __init__(
        self,
        dim: int,
        top_k: int,
        mode: str,
        num_heads: int = 4,
        retrieval_dropout: float = 0.0,
        use_gpu: bool = True,
    ):
        super().__init__()
        self.drug_reasoner = EntityRetrievalReasoner(
            dim=dim,
            top_k=top_k,
            mode=mode,
            num_heads=num_heads,
            retrieval_dropout=retrieval_dropout,
            use_gpu=use_gpu,
        )
        self.disease_reasoner = EntityRetrievalReasoner(
            dim=dim,
            top_k=top_k,
            mode=mode,
            num_heads=num_heads,
            retrieval_dropout=retrieval_dropout,
            use_gpu=use_gpu,
        )

    def build_indexes(self, drug_bank: torch.Tensor, disease_bank: torch.Tensor) -> None:
        self.drug_reasoner.build_index(drug_bank)
        self.disease_reasoner.build_index(disease_bank)

    def forward(
        self,
        drug_emb: torch.Tensor,
        disease_emb: torch.Tensor,
        drug_ids: torch.Tensor,
        disease_ids: torch.Tensor,
        drug_bank: torch.Tensor,
        disease_bank: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, PairRetrievalOutput]:
        drug_enriched, drug_info = self.drug_reasoner(drug_emb, drug_ids, drug_bank)
        disease_enriched, disease_info = self.disease_reasoner(disease_emb, disease_ids, disease_bank)
        return drug_enriched, disease_enriched, PairRetrievalOutput(drug=drug_info, disease=disease_info)
