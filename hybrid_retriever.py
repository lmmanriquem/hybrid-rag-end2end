"""
Hybrid BM25+DPR retriever for RAG-end2end training.

Extends RagRayDistributedRetriever: DPR fetches N_CANDIDATES passages,
BM25 re-ranks them, and scores are fused as:
    score(q, p) = alpha * BM25_norm(q, p) + (1 - alpha) * DPR_norm(q, p)

alpha=0.0 falls through to pure DPR with zero overhead.
"""
import logging

import numpy as np
from datasets import load_from_disk
from rank_bm25 import BM25Okapi
from transformers import RagConfig, RagTokenizer
from transformers.models.rag.retrieval_rag import CustomHFIndex

from distributed_ray_retriever import RagRayDistributedRetriever

logger = logging.getLogger(__name__)

N_CANDIDATES = 50  # DPR over-retrieval factor before BM25 re-ranking


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


class HybridRayDistributedRetriever(RagRayDistributedRetriever):
    """BM25+DPR hybrid retriever. Drop-in replacement for RagRayDistributedRetriever."""

    def __init__(
        self,
        config,
        question_encoder_tokenizer,
        generator_tokenizer,
        retrieval_workers,
        index=None,
        alpha: float = 0.5,
    ):
        super().__init__(
            config,
            question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer,
            retrieval_workers=retrieval_workers,
            index=index,
        )
        self.alpha = alpha
        self._query_texts = None

        if alpha > 0.0:
            logger.info("Building BM25 index from %s ...", config.passages_path)
            dataset = load_from_disk(config.passages_path)
            texts = dataset["text"]
            tokenized = [t.lower().split() for t in texts]
            self.bm25 = BM25Okapi(tokenized)
            logger.info("BM25 index ready (%d passages).", len(texts))
        else:
            self.bm25 = None

    # ------------------------------------------------------------------
    # Called from finetune_rag._step() and _generative_step() so that
    # both the training forward pass and validation generation use hybrid
    # retrieval consistently.
    # ------------------------------------------------------------------
    def set_query_texts(self, texts):
        self._query_texts = texts

    def retrieve(self, question_hidden_states: np.ndarray, n_docs: int):
        if self.alpha == 0.0 or self._query_texts is None or self.bm25 is None:
            return super().retrieve(question_hidden_states, n_docs)

        # Single-device MPS path: no Ray workers, call index directly.
        n_cand = max(n_docs * 10, N_CANDIDATES)
        doc_ids, retrieved_doc_embeds = self._main_retrieve(question_hidden_states, n_cand)
        doc_dicts = self.index.get_doc_dicts(doc_ids)

        out_ids, out_embeds, out_dicts = [], [], []

        for i in range(question_hidden_states.shape[0]):
            q_text = self._query_texts[i] if i < len(self._query_texts) else ""
            cand_ids = doc_ids[i]               # (n_cand,)
            cand_embeds = retrieved_doc_embeds[i]  # (n_cand, dim)

            # DPR scores: inner product of candidate embeddings with query vector
            dpr_scores = np.dot(cand_embeds, question_hidden_states[i]).astype(np.float32)

            # BM25 scores: fetch full-corpus scores, index by candidate ids
            all_bm25 = self.bm25.get_scores(q_text.lower().split())
            bm25_scores = all_bm25[cand_ids].astype(np.float32)

            fused = self.alpha * _minmax(bm25_scores) + (1.0 - self.alpha) * _minmax(dpr_scores)
            top_k = np.argsort(fused)[::-1][:n_docs]

            out_ids.append(cand_ids[top_k])
            out_embeds.append(cand_embeds[top_k])
            out_dicts.append({k: [v[j] for j in top_k] for k, v in doc_dicts[i].items()})

        return np.array(out_embeds), np.array(out_ids), out_dicts

    @classmethod
    def from_pretrained(
        cls,
        retriever_name_or_path,
        actor_handles,
        alpha: float = 0.5,
        indexed_dataset=None,
        **kwargs,
    ):
        config = kwargs.pop("config", None) or RagConfig.from_pretrained(
            retriever_name_or_path, **kwargs
        )
        rag_tokenizer = RagTokenizer.from_pretrained(retriever_name_or_path, config=config)
        question_encoder_tokenizer = rag_tokenizer.question_encoder
        generator_tokenizer = rag_tokenizer.generator

        if indexed_dataset is not None:
            config.index_name = "custom"
            index = CustomHFIndex(config.retrieval_vector_size, indexed_dataset)
        else:
            index = cls._build_index(config)

        return cls(
            config,
            question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer,
            retrieval_workers=actor_handles,
            index=index,
            alpha=alpha,
        )
