# Hybrid-RAG-end2end: BM25+DPR Hybrid Retrieval in the RAG Training Loop

**Paper:** coming soon

> **Based on:** Siriwardhana et al., *Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain Question Answering*, TACL 2023. ([ACL Anthology](https://aclanthology.org/2023.tacl-1.1/))

---

## What is this?

This repository implements **Hybrid-RAG-end2end**, a research extension of the RAG-end2end framework (Siriwardhana et al., TACL 2023) that integrates hybrid BM25+DPR retrieval directly into the training loop of a generative RAG system.

The original RAG-end2end trains a DPR retriever and a BART generator end-to-end, updating the FAISS knowledge base index periodically during training. This work extends that framework by replacing the pure DPR retriever with a hybrid retriever that fuses sparse (BM25) and dense (DPR) scores at every training step — not just at inference time.

The central hypothesis is that this hybrid signal helps the model adapt to specialized conversational domains (QAConv) where DPR pre-training coverage is weaker, while having a smaller or neutral effect on general encyclopedic domains (SQuAD).

---

## Repository Lineage

This repository is the third step in a chain of progressively extended codebases:

| Step | Repository | What it adds |
|---|---|---|
| 1 | [huggingface/transformers-research-projects](https://github.com/huggingface/transformers-research-projects/tree/main/rag-end2end-retriever) | Original RAG-end2end code by Siriwardhana et al. — NVIDIA/CUDA only |
| 2 | [lmmanriquem/rag-end2end-retriever](https://github.com/lmmanriquem/rag-end2end-retriever) | **Apple Silicon adaptation** — makes RAG-end2end fully functional on MPS (M1/M2/M3/M4) without any NVIDIA hardware |
| 3 | [lmmanriquem/hybrid-rag-end2end](https://github.com/lmmanriquem/hybrid-rag-end2end) *(this repo)* | **Hybrid retrieval contribution** — adds BM25+DPR fusion into the training loop on top of Step 2 |

**Step 2 is a standalone contribution.** Getting RAG-end2end to run on Apple Silicon required resolving non-trivial platform-specific issues: dual OpenMP conflicts between PyTorch MPS and FAISS, MPS dtype incompatibilities, macOS `spawn`-based multiprocessing failures in the FAISS re-encoding cycle, and CUDA-specific imports scattered throughout the codebase. The result — a fully functional RAG-end2end on M-series hardware — is independently useful for anyone who wants to reproduce or extend Siriwardhana et al. without NVIDIA GPUs.

If you only need RAG-end2end on Apple Silicon (no hybrid retrieval), use [lmmanriquem/rag-end2end-retriever](https://github.com/lmmanriquem/rag-end2end-retriever) directly.

---

## Research Contribution

**The fusion formula:**

```
score(q, p) = α · BM25̃(q, p) + (1 − α) · DPR̃(q, p)
```

Both scores are min-max normalized to [0, 1] before fusion. α ∈ {0.0, 0.3, 0.5, 0.7} is the balance parameter. Setting α = 0.0 recovers the original RAG-end2end baseline exactly.

**Gradient flow is unchanged:** BM25 is non-differentiable, but it operates only on passage selection (re-ranking). Gradients still flow exclusively through the DPR question encoder and the BART generator — the end-to-end training dynamic is preserved.

**How to use:**

```bash
# Baseline — pure DPR (reproduces Siriwardhana et al.)
python finetune_rag.py ... --alpha 0.0

# Hybrid BM25+DPR (this work)
python finetune_rag.py ... --alpha 0.3
```

The only required change from the baseline run is adding `--alpha`. All other arguments remain identical.

---

## Hypotheses

| | Hypothesis |
|---|---|
| **H1** | BM25+DPR hybrid improves EM and F1 on QAConv vs. pure DPR baseline (α = 0.0) |
| **H2** | The improvement is larger in QAConv (specialized conversational domain) than in SQuAD (general encyclopedic) |
| **H3** | The optimal α differs by domain: QAConv favors higher BM25 weight than SQuAD |

---

## Results

Experiments run on a 10% random subset of each training set (SQuAD: 8,760 examples; QAConv: 2,600 examples) with full knowledge bases retained. Exact Match (EM) is reported as the primary metric.

| Experiment | Dataset | α | Best EM | vs. Baseline | Status |
|---|---|---|---|---|---|
| Baseline (DPR) | SQuAD 10% | 0.0 | 0.3300 | — | ✅ Done |
| Hybrid (BM25+DPR) | SQuAD 10% | 0.7 | 0.3933 | +19.2% | ✅ Done |
| Baseline (DPR) | QAConv 10% | 0.0 | 0.0867 | — | ✅ Done |
| Hybrid (BM25+DPR) | QAConv 10% | 0.7 | 0.1067 | +22.9% | ✅ Done |
| Paper target (Siriwardhana et al.) | SQuAD full | — | 40.02 | — | — |
| Paper target (Siriwardhana et al.) | QAConv full | — | 24.25 | — | — |

---

## Citation

```bibtex
@article{manrique2026hybridrag,
  title     = {Hybrid-RAG-end2end: BM25+DPR Hybrid Retrieval in the RAG Training Loop},
  author    = {Manrique, Luis Manuel},
  year      = {2026},
  note      = {Manuscript in preparation}
}
```

---

## Experiment Hardware

| Component | Specification |
|---|---|
| Machine | MacBook Pro M4 Max |
| Unified Memory | 48 GB |
| GPU Cores | 40 (Apple Silicon GPU) |
| CPU Cores | 16 (12 performance + 4 efficiency) |
| Architecture | arm64 (Apple Silicon) |
| OS | macOS Tahoe 26.3.1 |
| GPU Backend | MPS (Metal Performance Shaders) via PyTorch |
| Re-encoding | CPU (MPS is occupied by the training loop) |

> The codebase also fully preserves the original NVIDIA/CUDA code path. No changes are required to run on NVIDIA hardware — see [NVIDIA / CUDA Compatibility](#nvidia--cuda-compatibility).

---

## Datasets

| Dataset | Domain | QA pairs | KB passages | Paper table |
|---|---|---|---|---|
| SQuAD v1.1 | General (Wikipedia) | ~87K train | ~35K | Table 5, §5.3 |
| QAConv v1.1 | Conversational (emails, Slack, papers) | ~26K train | ~69K | Table 1 |

Full dataset preparation, FAISS index build, and experiment commands are documented in [EXPERIMENTS.md](./EXPERIMENTS.md).

---

## Setup

### Prerequisites

- macOS with Apple Silicon (M1 / M2 / M3 / M4)
- [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) installed
- Internet connection (model downloads ~2.5 GB on first run)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/lmmanriquem/hybrid-rag-end2end.git
cd hybrid-rag-end2end
```

---

### Step 2 — Create a Python 3.11 conda environment

> **Important:** Python 3.13+ is NOT compatible with the dependency stack. Use Python 3.11.

```bash
conda create -n hybrid-rag-env python=3.11
conda activate hybrid-rag-env
python --version   # must show Python 3.11.x
```

> **VS Code users:** VS Code auto-activates the `venv/` folder when you open a terminal, overriding conda. If your prompt shows both `(hybrid-rag-env)` and `(venv)`, run `deactivate` first, then `conda activate hybrid-rag-env`.

---

### Step 3 — Install dependencies

```bash
python setup_env.py
```

This script auto-detects Apple Silicon, installs PyTorch with MPS support, and skips NVIDIA-only packages. Expected output at the end:

```
✓ MPS  available  — Apple Silicon GPU will be used for training
── Setup complete ───────────────────────────────────────────
```

---

### Step 4 — Pin dependency versions

```bash
pip install "transformers>=4.30.0,<5.0.0" "datasets>=2.10.0,<3.0.0"
```

---

### Step 5 — Verify installation

```bash
python -c "
import torch, pytorch_lightning as pl, transformers, datasets, faiss, ray
from rank_bm25 import BM25Okapi
print(f'torch:             {torch.__version__}')
print(f'MPS available:     {torch.backends.mps.is_available()}')
print(f'pytorch-lightning: {pl.__version__}')
print(f'transformers:      {transformers.__version__}')
print(f'datasets:          {datasets.__version__}')
from transformers import RagSequenceForGeneration, RagTokenForGeneration, RagRetriever
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
print('RAG classes:       OK')
print('rank_bm25:         OK')
"
```

All lines must print without errors. `MPS available` must be `True`.

---

### Step 6 — Prepare training data

Training data must be plain text files, one item per line:

```
data/
  train.source   ← one question per line
  train.target   ← one answer per line (same order as .source)
  val.source / val.target / test.source / test.target
```

Example — smoke test with dummy data:

```bash
mkdir -p smoke_test/data smoke_test/kb smoke_test/output smoke_test/shards

python - << 'EOF'
questions = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water?",
    "How many planets are in the solar system?",
    "What language do Brazilians speak?",
]
answers = ["Paris", "Shakespeare", "100 degrees Celsius", "eight", "Portuguese"]

for split, q, a in [("train", questions, answers),
                    ("val",   questions[:2], answers[:2]),
                    ("test",  questions[:2], answers[:2])]:
    open(f"smoke_test/data/{split}.source","w").write("\n".join(q)+"\n")
    open(f"smoke_test/data/{split}.target","w").write("\n".join(a)+"\n")

passages = [
    ("France",       "Paris is the capital and largest city of France."),
    ("Shakespeare",  "William Shakespeare wrote Romeo and Juliet around 1594."),
    ("Water",        "Water boils at 100 degrees Celsius at sea level."),
    ("Solar System", "The solar system has eight planets."),
    ("Brazil",       "Portuguese is the official language of Brazil."),
    ("Science",      "Physics studies matter energy and the fundamental forces of nature."),
    ("History",      "The Roman Empire fell in 476 AD."),
    ("Mathematics",  "The square root of 144 is 12."),
]
with open("smoke_test/kb/passages.tsv","w") as f:
    for title, text in passages:
        f.write(f"{title}\t{text}\n")

print("(GOOD) Data files created. OK")
EOF
```

---

### Step 7 — Encode the knowledge base

Encode passages with DPR (two separate steps to avoid the macOS dual-OpenMP issue — see [Known Issues](#known-issues-on-macos-arm64)):

```bash
python - << 'EOF'
import torch, numpy as np
from datasets import Dataset
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Encoding on: {device}")

titles, texts = [], []
with open("smoke_test/kb/passages.tsv") as f:
    for line in f:
        parts = line.strip().split("\t", 1)
        if len(parts) == 2:
            titles.append(parts[0]); texts.append(parts[1])

ctx_encoder   = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base").to(device)
ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
ctx_encoder.eval()

all_emb = []
with torch.no_grad():
    for i in range(0, len(titles), 4):
        inp = ctx_tokenizer(titles[i:i+4], texts[i:i+4],
                            truncation=True, padding="longest",
                            max_length=512, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        all_emb.append(ctx_encoder(**inp, return_dict=True).pooler_output.cpu().float().numpy())

import numpy as np
embeddings = np.concatenate(all_emb)
ds = Dataset.from_dict({"title": titles, "text": texts,
                        "embeddings": [e for e in embeddings]})
ds.save_to_disk("smoke_test/kb/my_knowledge_dataset")
print("(GOOD) Dataset with embeddings saved. OK")
EOF
```

```bash
KMP_DUPLICATE_LIB_OK=TRUE python - << 'EOF'
import faiss
from datasets import load_from_disk

faiss.omp_set_num_threads(1)
ds = load_from_disk("smoke_test/kb/my_knowledge_dataset")
index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
ds.add_faiss_index("embeddings", custom_index=index)
ds.get_index("embeddings").save("smoke_test/kb/my_knowledge_dataset_hnsw_index.faiss")
print("(GOOD) FAISS index saved. OK")
EOF
```

---

### Step 8 — Start the Ray cluster

```bash
ray start --head
```

---

### Step 9 — Run the smoke test

Verify the pipeline end-to-end with 1 training batch + 1 validation batch:

```bash
KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false python finetune_rag.py \
    --data_dir              smoke_test/data \
    --output_dir            smoke_test/output \
    --model_name_or_path    facebook/rag-token-base \
    --model_type            rag_token \
    --accelerator           mps \
    --devices               1 \
    --precision             32 \
    --do_train \
    --end2end \
    --n_val                 -1 \
    --train_batch_size      2 \
    --eval_batch_size       1 \
    --max_source_length     128 \
    --max_target_length     25 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing       0.1 \
    --dropout               0.1 \
    --attention_dropout     0.1 \
    --weight_decay          0.001 \
    --adam_epsilon          1e-08 \
    --max_grad_norm         0.1 \
    --lr_scheduler          polynomial \
    --learning_rate         3e-05 \
    --num_train_epochs      1 \
    --warmup_steps          0 \
    --gradient_accumulation_steps 1 \
    --distributed_retriever ray \
    --num_retrieval_workers 1 \
    --passages_path         smoke_test/kb/my_knowledge_dataset \
    --index_path            smoke_test/kb/my_knowledge_dataset_hnsw_index.faiss \
    --index_name            custom \
    --context_encoder_name  facebook/dpr-ctx_encoder-multiset-base \
    --csv_path              smoke_test/kb/passages.tsv \
    --index_gpus            1 \
    --gpu_order             "[]" \
    --shard_dir             smoke_test/shards \
    --indexing_freq         500 \
    --num_workers           0 \
    --alpha                 0.0 \
    --fast_dev_run
```

Expected: `Trainer.fit stopped: max_steps=1 reached.` with a numeric loss value.

To also verify the hybrid path, run the same command with `--alpha 0.3`. The loss value will differ slightly (different passages are retrieved), confirming BM25 fusion is active.

---

### Step 10 — Stop the Ray cluster

```bash
ray stop
```

---

## Experiments

Full step-by-step instructions for all experiments are in [EXPERIMENTS.md](./EXPERIMENTS.md), including dataset preparation, FAISS index builds, quick tests, trigger test, and full training commands for both baseline and hybrid configurations.

### Experiment Status

| Experiment | Dataset | α | Est. Time (M4 Max) | Status | EM |
|---|---|---|---|---|---|
| Smoke test | Dummy | 0.0 | < 1 min | ✅ Done | loss ≈ 76.5 |
| Smoke test | Dummy | 0.3 | < 1 min | ✅ Done | loss ≈ 81.4 |
| Quick test | SQuAD mini (500 / 2K KB) | 0.0 | ~1h45min | ✅ Done | 0.07 (best) |
| Quick test | QAConv mini (300 / 1.5K KB) | 0.0 | ~50min | ✅ Done | 0.22 (best) |
| FAISS index build | SQuAD full (34,620 passages) | — | ~2 min | ✅ Done | — |
| FAISS index build | QAConv full (68,707 passages) | — | ~6 min | ✅ Done | — |
| FAISS re-encoding trigger test | QAConv (1,100 / 3K KB) | — | ~10 min | ✅ Done | fired ✅ |
| **Baseline** | SQuAD full (~87K / 35K KB) | 0.0 | ~4.5 days | ⏳ Pending | target: 40.02 |
| **Baseline** | QAConv full (~26K / 69K KB) | 0.0 | ~1.7 days | ⏳ Pending | target: 24.25 |
| **Mini-ablation** | SQuAD mini | 0.3 / 0.5 / 0.7 | ~5h total | ⏳ Pending | — |
| **Mini-ablation** | QAConv mini | 0.3 / 0.5 / 0.7 | ~2.5h total | ⏳ Pending | — |
| **Hybrid (best α)** | SQuAD full | TBD | ~4.5 days | ⏳ Pending | — |
| **Hybrid (best α)** | QAConv full | TBD | ~1.7 days | ⏳ Pending | — |

> ⚠️ Full training requires `--val_check_interval 500`. Without it, validation runs after every training batch and SQuAD full would take ~6,555 days. See [EXPERIMENTS.md](./EXPERIMENTS.md) for the full explanation.

---

## Daily Workflow (returning sessions)

Steps 1–9 are one-time setup. Once the smoke test has passed, this is all you need each session:

### Starting a session

```bash
deactivate                    # if VS Code auto-activated a venv
conda activate hybrid-rag-env
ray start --head
```

### Running the smoke test

The KB and FAISS index from first setup persist on disk:

```bash
KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false python finetune_rag.py \
    --data_dir              smoke_test/data \
    --output_dir            smoke_test/output \
    --model_name_or_path    facebook/rag-token-base \
    --model_type            rag_token \
    --accelerator           mps \
    --devices               1 \
    --precision             32 \
    --do_train \
    --end2end \
    --n_val                 -1 \
    --train_batch_size      2 \
    --eval_batch_size       1 \
    --max_source_length     128 \
    --max_target_length     25 \
    --val_max_target_length 25 \
    --test_max_target_length 25 \
    --label_smoothing       0.1 \
    --dropout               0.1 \
    --attention_dropout     0.1 \
    --weight_decay          0.001 \
    --adam_epsilon          1e-08 \
    --max_grad_norm         0.1 \
    --lr_scheduler          polynomial \
    --learning_rate         3e-05 \
    --num_train_epochs      1 \
    --warmup_steps          0 \
    --gradient_accumulation_steps 1 \
    --distributed_retriever ray \
    --num_retrieval_workers 1 \
    --passages_path         smoke_test/kb/my_knowledge_dataset \
    --index_path            smoke_test/kb/my_knowledge_dataset_hnsw_index.faiss \
    --index_name            custom \
    --context_encoder_name  facebook/dpr-ctx_encoder-multiset-base \
    --csv_path              smoke_test/kb/passages.tsv \
    --index_gpus            1 \
    --gpu_order             "[]" \
    --shard_dir             smoke_test/shards \
    --indexing_freq         500 \
    --num_workers           0 \
    --alpha                 0.0 \
    --fast_dev_run
```

### What persists between sessions

| Resource | Persists? | Notes |
|---|---|---|
| `hybrid-rag-env` conda environment | ✓ Yes | No need to reinstall |
| Downloaded models (HuggingFace cache) | ✓ Yes | Cached in `~/.cache/huggingface/` |
| `smoke_test/kb/` (dataset + FAISS index) | ✓ Yes | Already encoded, ready to use |
| Ray cluster | ✗ No | Must run `ray start --head` each session |

### Ending a session

```bash
ray stop
```

---

## Environment Variables — macOS Required

These must be set before any run that combines PyTorch + FAISS on macOS:

| Variable | Value | Reason |
|---|---|---|
| `KMP_DUPLICATE_LIB_OK` | `TRUE` | PyTorch and FAISS both ship `libomp.dylib`; the second load aborts the process without this |
| `TOKENIZERS_PARALLELISM` | `false` | HuggingFace tokenizers conflict with macOS `spawn`-based multiprocessing during KB re-encoding |

Both are already exported in `finetune_rag_mps_end2end.sh`.

---

## Known Issues on macOS ARM64

| Issue | Cause | Fix applied |
|---|---|---|
| `zsh: segmentation fault` during KB encoding | `dataset.map()` + tokenizer parallelism on macOS | Custom encoding loop in Step 7 |
| `OMP: Error #15` / abort during FAISS index build | Dual `libomp.dylib` (PyTorch + FAISS) | Separate FAISS step + `KMP_DUPLICATE_LIB_OK=TRUE` |
| `ImportError: cannot import name 'AdamW' from 'transformers'` | `AdamW` removed from transformers in 4.x | Import from `torch.optim` in `lightning_base.py` |
| `--fp16` crash on MPS | APEX not available on Apple Silicon | fp16 blocked in `lightning_base.py`; use `--precision 32` or `bf16-mixed` |
| Segfault during first training batch | Dual OpenMP runtimes (PyTorch MPS + FAISS CPU) race during HNSW search | `faiss.omp_set_num_threads(1)` at startup in `finetune_rag.py` (macOS only) |
| Segfault + `21 leaked semaphore objects` in Epoch 0 | DataLoader spawns child processes; MPS inaccessible from child processes on macOS | `--num_workers 0` required on all Apple Silicon runs |
| `TypeError: Cannot convert a MPS Tensor to float64 dtype` | PyTorch Lightning passes Python/numpy scalars to `log_dict` as float64; MPS doesn't support float64 | Explicit `torch.float32` cast in `validation_epoch_end` in `finetune_rag.py` |
| `RuntimeError: size of tensor a must match tensor b` during FAISS build | Some passages exceed 512 tokens; truncation not triggered without `max_length` | Added `max_length=512` in `use_own_knowledge_dataset.py` |

---

## How FAISS Re-encoding Works on Apple Silicon

RAG-end2end re-encodes the entire knowledge base every `--indexing_freq` batches using the updated DPR context encoder weights. This is what makes training truly end-to-end: the retriever adapts alongside the generator.

On Apple Silicon the M-series CPU handles re-encoding (MPS is occupied by training). Re-encoding runs as a background child process and does not block training steps.

```python
if torch.cuda.is_available():
    free_gpu_list = ["cuda:0", "cuda:1", ...]   # NVIDIA: use free GPUs
else:
    free_gpu_list = ["cpu"] * index_gpus        # Apple Silicon: use CPU
```

| Component | NVIDIA | Apple Silicon |
|---|---|---|
| Generator (BART) | ✅ End-to-end | ✅ End-to-end |
| Question encoder (DPR) | ✅ Updated | ✅ Updated |
| Context encoder (DPR) | ✅ Reflected in FAISS | ✅ Reflected in FAISS |
| Re-encoding device | NVIDIA GPU | M-series CPU (~6 min/cycle) |

---

## NVIDIA / CUDA Compatibility

The original CUDA code path is fully preserved. On NVIDIA hardware:

- `pynvml` is loaded lazily at runtime only when CUDA is detected
- `nvidia-ml-py3` is installed by `setup_env.py` only on NVIDIA systems
- DDP multi-GPU strategy activates automatically when `--gpus > 1`
- FAISS uses all available CPU cores for indexing

No changes are required to run on NVIDIA hardware.

---

## Implementation Notes

This repository extends [lmmanriquem/rag-end2end-retriever](https://github.com/lmmanriquem/rag-end2end-retriever) (the Apple Silicon adaptation of the original Siriwardhana et al. codebase) with one additional category of changes:

**Apple Silicon adaptations** are already present in the base repo — see [lmmanriquem/rag-end2end-retriever](https://github.com/lmmanriquem/rag-end2end-retriever) for the full list of platform fixes (FAISS OpenMP conflict, MPS dtype handling, multiprocessing fixes for macOS `spawn`, CUDA-specific import removal).

**Hybrid retrieval contribution** (in `hybrid_retriever.py`, `build_bm25_index.py`, and additions to `finetune_rag.py`): `HybridRayDistributedRetriever` subclasses `RagRayDistributedRetriever` and overrides `retrieve()` to fuse BM25 and DPR scores before returning the top-K passages to the training loop. The `--alpha` argument controls the fusion weight. Setting `--alpha 0.0` disables BM25 entirely with zero overhead.

---

## References

- Siriwardhana et al. (2023). *Improving the Domain Adaptation of RAG Models for Open Domain QA.* TACL. [ACL Anthology](https://aclanthology.org/2023.tacl-1.1/)
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS. [ACM](https://dl.acm.org/doi/abs/10.5555/3495724.3496517)
- Karpukhin et al. (2020). *Dense Passage Retrieval for Open-Domain Question Answering.* EMNLP.
- Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval.
