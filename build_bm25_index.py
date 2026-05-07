"""Utility script to verify and time BM25 index construction for a passages dataset."""
import argparse
import time

from datasets import load_from_disk
from rank_bm25 import BM25Okapi


def build_bm25_index(passages_path: str):
    print(f"Loading passages from {passages_path} ...")
    dataset = load_from_disk(passages_path)
    texts = dataset["text"]
    print(f"  {len(texts)} passages loaded.")

    print("Tokenizing and building BM25Okapi index ...")
    t0 = time.time()
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s.")

    sample_query = tokenized[0][:3]
    scores = bm25.get_scores(sample_query)
    print(f"  Sample query '{' '.join(sample_query)}': top-1 score = {scores.max():.4f}")
    return bm25


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and time a BM25 index for a HuggingFace passages dataset."
    )
    parser.add_argument(
        "--passages_path",
        required=True,
        help="Path to the HuggingFace dataset (output of use_own_knowledge_dataset.py).",
    )
    args = parser.parse_args()
    build_bm25_index(args.passages_path)
