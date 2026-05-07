#!/usr/bin/env bash
# ============================================================
# run_mini_ablation.sh
# Phase 1 — Hybrid-RAG-end2end mini-ablation over α ∈ {0.3, 0.5, 0.7}
#
# Runs 6 sequential experiments (3 alphas × 2 datasets) on mini
# subsets to identify the best α before committing to full training.
#
# Estimated total time on M4 Max (MPS):
#   QAConv mini × 3: ~2.5h   (~50 min each)
#   SQuAD mini  × 3: ~5.25h  (~1h 45min each)
#   Total:           ~7.75h
#
# Output structure:
#   qaconv_mini/output_a{03,05,07}/metrics.json
#   squad_mini/output_a{03,05,07}/metrics.json
#
# Usage:
#   conda activate hybrid-rag-env
#   bash run_mini_ablation.sh 2>&1 | tee ablation_log.txt
# ============================================================

export KMP_DUPLICATE_LIB_OK=TRUE
export TOKENIZERS_PARALLELISM=false

ALPHAS=(0.3 0.5 0.7)
TAGS=(a03 a05 a07)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_experiment() {
    local DATASET_DIR="$1"
    local ALPHA="$2"
    local TAG="$3"
    local OUT="${DATASET_DIR}/output_${TAG}"
    local SHARDS="${DATASET_DIR}/shards_${TAG}"

    log "--- START | dataset=${DATASET_DIR} | α=${ALPHA} | output → ${OUT}"
    local START
    START=$(date +%s)

    mkdir -p "${OUT}" "${SHARDS}"

    python finetune_rag.py \
        --data_dir              "${DATASET_DIR}" \
        --output_dir            "${OUT}" \
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
        --passages_path         "${DATASET_DIR}/kb/my_knowledge_dataset" \
        --index_path            "${DATASET_DIR}/kb/my_knowledge_dataset_hnsw_index.faiss" \
        --index_name            custom \
        --context_encoder_name  facebook/dpr-ctx_encoder-multiset-base \
        --csv_path              "${DATASET_DIR}/kb/passages.tsv" \
        --index_gpus            1 \
        --gpu_order             "[]" \
        --shard_dir             "${SHARDS}" \
        --indexing_freq         500 \
        --num_workers           0 \
        --alpha                 "${ALPHA}"

    local END
    END=$(date +%s)
    local ELAPSED=$(( (END - START) / 60 ))
    log "--- DONE  | dataset=${DATASET_DIR} | α=${ALPHA} | ${ELAPSED} min elapsed"
}

# ── Ray ──────────────────────────────────────────────────────
log "Starting Ray cluster..."
ray stop 2>/dev/null || true
ray start --head
log "Ray ready."

# ── QAConv mini ablation (~2.5h) ─────────────────────────────
log "=== QAConv mini ablation START ==="
for i in "${!ALPHAS[@]}"; do
    run_experiment "qaconv_mini" "${ALPHAS[$i]}" "${TAGS[$i]}"
done
log "=== QAConv mini ablation DONE ==="

# ── SQuAD mini ablation (~5.25h) ─────────────────────────────
log "=== SQuAD mini ablation START ==="
for i in "${!ALPHAS[@]}"; do
    run_experiment "squad_mini" "${ALPHAS[$i]}" "${TAGS[$i]}"
done
log "=== SQuAD mini ablation DONE ==="

# ── Ray ──────────────────────────────────────────────────────
ray stop
log "Ray stopped."

# ── Results summary ──────────────────────────────────────────
log "=== MINI-ABLATION RESULTS ==="
python3 - <<'PYEOF'
import json, os, re
from datetime import datetime

RUNS = [
    ("qaconv_mini", "QAConv mini", "0.3", "a03"),
    ("qaconv_mini", "QAConv mini", "0.5", "a05"),
    ("qaconv_mini", "QAConv mini", "0.7", "a07"),
    ("squad_mini",  "SQuAD mini",  "0.3", "a03"),
    ("squad_mini",  "SQuAD mini",  "0.5", "a05"),
    ("squad_mini",  "SQuAD mini",  "0.7", "a07"),
]

# ── 1. Collect results ───────────────────────────────────────
rows = []
for dataset_dir, label, alpha, tag in RUNS:
    out = f"{dataset_dir}/output_{tag}"
    path = f"{out}/metrics.json"
    if not os.path.exists(path):
        rows.append({"dataset_dir": dataset_dir, "label": label, "alpha": alpha,
                     "tag": tag, "best_em": None, "final_em": None, "status": "NOT FOUND"})
        continue
    with open(path) as f:
        data = json.load(f)
    vals = data.get("val", [])
    if not vals:
        rows.append({"dataset_dir": dataset_dir, "label": label, "alpha": alpha,
                     "tag": tag, "best_em": None, "final_em": None, "status": "NO DATA"})
        continue
    best_em = max(v.get("val_avg_em", 0) for v in vals)
    final_em = vals[-1].get("val_avg_em", 0)
    rows.append({"dataset_dir": dataset_dir, "label": label, "alpha": alpha,
                 "tag": tag, "best_em": best_em, "final_em": final_em, "status": "ok"})

# ── 2. Print terminal table ───────────────────────────────────
print()
print(f"{'Dataset':<14} {'α':<6} {'Best EM':<10} {'Final EM':<10} {'Output dir'}")
print("-" * 62)
for r in rows:
    if r["status"] == "ok":
        out = f"{r['dataset_dir']}/output_{r['tag']}/"
        print(f"{r['label']:<14} {r['alpha']:<6} {r['best_em']:<10.4f} {r['final_em']:<10.4f} {out}")
    else:
        print(f"{r['label']:<14} {r['alpha']:<6} {r['status']}")

best_per_dataset = {}
for ds in [("qaconv_mini", "QAConv"), ("squad_mini", "SQuAD")]:
    group = [r for r in rows if r["dataset_dir"] == ds[0] and r["status"] == "ok"]
    if group:
        winner = max(group, key=lambda r: r["best_em"])
        best_per_dataset[ds[0]] = winner
        print(f"\nBest α for {ds[1]}: {winner['alpha']}  (Best EM = {winner['best_em']:.4f})")

# ── 3. Write ablation_summary.json ───────────────────────────
summary = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "runs": [
        {
            "dataset": r["dataset_dir"],
            "label": r["label"],
            "alpha": float(r["alpha"]),
            "best_em": round(r["best_em"], 6) if r["best_em"] is not None else None,
            "final_em": round(r["final_em"], 6) if r["final_em"] is not None else None,
            "output_dir": f"{r['dataset_dir']}/output_{r['tag']}",
            "status": r["status"],
        }
        for r in rows
    ],
    "best_alpha": {
        ds: {"alpha": float(w["alpha"]), "best_em": round(w["best_em"], 6)}
        for ds, w in best_per_dataset.items()
    },
}
with open("ablation_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nablation_summary.json written.")

# ── 4. Auto-update EXPERIMENTS.md results table ──────────────
experiments_path = "EXPERIMENTS.md"
if not os.path.exists(experiments_path):
    print("EXPERIMENTS.md not found — skipping auto-update.")
else:
    with open(experiments_path) as f:
        content = f.read()

    # Replace each pending row with actual values
    label_map = {"qaconv_mini": "QAConv mini", "squad_mini": "SQuAD mini"}
    for r in rows:
        label = label_map[r["dataset_dir"]]
        alpha = r["alpha"]
        if r["status"] == "ok":
            best_str  = f"{r['best_em']:.4f}"
            final_str = f"{r['final_em']:.4f}"
            note = "✅"
        else:
            best_str = final_str = "—"
            note = r["status"]
        old = f"| {label} | {alpha} | — | — | ⏳ Pending |"
        new = f"| {label} | {alpha} | {best_str} | {final_str} | {note} |"
        content = content.replace(old, new)

    # Replace the "Best α selected" line
    qaconv_winner = best_per_dataset.get("qaconv_mini")
    squad_winner  = best_per_dataset.get("squad_mini")
    qaconv_str = f"α={qaconv_winner['alpha']} (EM={qaconv_winner['best_em']:.4f})" if qaconv_winner else "⏳ pending"
    squad_str  = f"α={squad_winner['alpha']} (EM={squad_winner['best_em']:.4f})"  if squad_winner  else "⏳ pending"
    old_line = "**Best α selected:** QAConv → ⏳ pending | SQuAD → ⏳ pending"
    new_line = f"**Best α selected:** QAConv → {qaconv_str} | SQuAD → {squad_str}"
    content = content.replace(old_line, new_line)

    with open(experiments_path, "w") as f:
        f.write(content)
    print("EXPERIMENTS.md results table updated.")

print()
PYEOF

log "Done."
log "  ablation_summary.json — consolidated results (JSON)"
log "  EXPERIMENTS.md        — results table updated in place"
log "  Full training commands are in EXPERIMENTS.md — Hybrid Experiments section."
