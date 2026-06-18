#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CR-HKGE final study (rewritten for fair, multi-seed comparison)
#
# Two correctness fixes vs the original script:
#
# 1. CONSISTENT ALPHA + SINGLE-FLAG ABLATIONS.
#    The headline full model and every ablation now share ONE canonical flag
#    stem (`cr_full_stem`). Each ablation is that stem with EXACTLY ONE flag
#    flipped (appended override; argparse uses the last value). This removes the
#    original confounds where:
#      - the full model ran at alpha=0.5 although the paper reports alpha=0.1,
#      - A_no_relation_attention also silently set relation_aware_message=0,
#      - A_no_fragrance_prior / A_no_relation_message ran at alpha=0.5.
#    Alpha is fixed at 0.1 for ALL variants (override with CR_ALPHA=...).
#    cr_relation_aware_message stays 1 everywhere except A_no_relation_message.
#
# 2. MULTI-SEED. Every target is trained+evaluated once per seed in $SEEDS, and
#    the results are aggregated to mean +/- std so significance is assessable.
#
# The exact flag that differs per variant is documented in IMPLEMENTATION_NOTES.md.
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/Model"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/final_study/logs}"
DATASET="${DATASET:-dataset-aromatique-kgat-ready}"

# Shared, configurable cross-reference strength. 0.1 = paper's final model.
CR_ALPHA="${CR_ALPHA:-0.1}"
# Seeds to average over. Configurable via env var.
SEEDS="${SEEDS:-2019 2020 2021 2022 2023}"

mkdir -p "$LOG_DIR"

COMMON_ARGS=(
  --data_path ../
  --dataset "$DATASET"
  --alg_type bi
  --adj_type si
  --adj_uni_type sum
  --regs '[1e-5,1e-5]'
  --layer_size '[64,32,16]'
  --embed_size 64
  --kge_size 64
  --lr 0.0001
  --batch_size 64
  --mess_dropout '[0.1,0.1,0.1]'
  --node_dropout '[0.1]'
  --Ks '[3,5,10]'
  --epoch "${CR_HKGE_FINAL_EPOCHS:-100}"
  --save_flag 1
)

# Canonical full CR-HKGE flag stem. ALL CR-HKGE variants below are built from
# this exact string with at most one overriding flag appended, so every
# comparison differs by a single component only.
cr_full_stem() {
  echo "--model_type cr_hkge \
--cr_use_relation_weight 1 \
--cr_use_cross_ref 1 \
--cr_relation_weight_mode semantic \
--cr_relation_prior_mode fragrance \
--cr_relation_prior_strength 1.0 \
--cr_relation_attention_scale type_count \
--cr_relation_aware_message 1 \
--cr_relation_message_scale type_count \
--cr_cross_ref_bi_interaction 0 \
--cr_cross_ref_gate 0 \
--cr_cross_ref_alpha ${CR_ALPHA} \
--cr_best_metric ndcg --cr_best_k 3"
}

model_args() {
  case "$1" in
    bprmf)
      echo "--model_type bprmf"
      ;;
    cke)
      echo "--model_type cke"
      ;;
    nfm)
      echo "--model_type nfm"
      ;;
    cfkg)
      # CFKG now defaults to Adam (parser default --cfkg_optimizer adam), matching
      # every other model. Pass --cfkg_optimizer sgd to reproduce the legacy result.
      echo "--model_type cfkg"
      ;;
    kgat)
      echo "--model_type kgat"
      ;;
    cr_hkge_final)
      # Headline model: full CR-HKGE stem, alpha=$CR_ALPHA (0.1 by default).
      echo "$(cr_full_stem) --cr_export_embeddings 1 --cr_model_version cr_hkge_final_flow"
      ;;
    A_no_cross_reference)
      # Ablation -> differs by ONE flag: cr_use_cross_ref (1 -> 0).
      echo "$(cr_full_stem) --cr_use_cross_ref 0"
      ;;
    A_no_relation_attention)
      # Ablation -> differs by ONE flag: cr_use_relation_weight (1 -> 0).
      # relation_aware_message stays 1 (held constant); it is a sub-feature of
      # relation attention and is inert at the model level when weights are off.
      echo "$(cr_full_stem) --cr_use_relation_weight 0"
      ;;
    A_no_relation_message)
      # Ablation -> differs by ONE flag: cr_relation_aware_message (1 -> 0).
      # This is the ONLY variant that touches relation_aware_message.
      echo "$(cr_full_stem) --cr_relation_aware_message 0"
      ;;
    A_no_fragrance_prior)
      # Ablation -> differs by ONE flag: cr_relation_prior_mode (fragrance -> none).
      echo "$(cr_full_stem) --cr_relation_prior_mode none"
      ;;
    A_no_novelty_modules)
      # Control (NOT a single-flag ablation, by design): all CR-HKGE novelties off,
      # i.e. KGAT wrapped in the CR-HKGE code path. Three flags differ on purpose.
      echo "$(cr_full_stem) --cr_use_relation_weight 0 --cr_use_cross_ref 0 --cr_relation_aware_message 0"
      ;;
    *)
      echo "unknown study target: $1" >&2
      return 1
      ;;
  esac
}

run_target() {
  local target="$1"
  read -r -a extra_args <<< "$(model_args "$target")"

  for seed in $SEEDS; do
    local tag="${target}_seed${seed}"
    local train_log="$LOG_DIR/${tag}_train.log"
    local eval_log="$LOG_DIR/${tag}_subset_eval.log"
    # Per-seed weights dir so checkpoints never collide between seeds.
    local weights_path="../final_study/${target}/seed_${seed}/"

    echo "==> Training $target (seed=$seed)"
    (
      cd "$MODEL_DIR"
      python Main.py \
        --weights_path "$weights_path" \
        --seed "$seed" \
        "${COMMON_ARGS[@]}" \
        "${extra_args[@]}"
    ) 2>&1 | tee "$train_log"

    echo "==> Evaluating $target (seed=$seed)"
    (
      cd "$MODEL_DIR"
      python evaluate_item_subsets.py \
        --weights_path "$weights_path" \
        --seed "$seed" \
        "${COMMON_ARGS[@]}" \
        "${extra_args[@]}"
    ) 2>&1 | tee "$eval_log"

    echo "==> Finished $target (seed=$seed)"
  done
}

if [ "$#" -eq 0 ]; then
  targets=(
    bprmf
    cke
    nfm
    cfkg
    kgat
    cr_hkge_final
    A_no_cross_reference
    A_no_relation_attention
    A_no_relation_message
    A_no_fragrance_prior
    A_no_novelty_modules
  )
else
  targets=("$@")
fi

for target in "${targets[@]}"; do
  run_target "$target"
done

# Per-seed flat table (one row per target+seed) for inspection ...
python "$ROOT_DIR/scripts/summarize_final_study.py" "$LOG_DIR"/*_subset_eval.log \
  | tee "$LOG_DIR/final_summary_per_seed.md"

# ... and the aggregated mean +/- std table across seeds (the headline result).
python "$ROOT_DIR/scripts/summarize_multiseed_study.py" \
  --csv "$LOG_DIR/final_summary_meanstd.csv" \
  "$LOG_DIR"/*_subset_eval.log \
  | tee "$LOG_DIR/final_summary_meanstd.md"
