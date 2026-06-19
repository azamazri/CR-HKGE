#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CR-HKGE Plan B (gated conditional enrichment) hold-out 2x2 study
#
# Identical protocol to run_cr_hkge_holdout_study.sh, but the CR-HKGE cells use
# Plan B: STANDARD products (no inspired_by) get EXACTLY the plain-KGAT update,
# so CR-HKGE can no longer underperform KGAT on standard products, while enriched
# products keep the full CR-HKGE treatment (--cr_planB_gate 1).
#
#                       | full KG (control)   | hold-out KG (direct paths removed)
#   --------------------+---------------------+-----------------------------------
#   KGAT                | kgat_fullkg         | kgat_holdout
#   CR-HKGE-PlanB (a=.1)| crhkgeB_fullkg      | crhkgeB_holdout
#
# Every cell is evaluated on overall / enriched / standard subsets, with the
# enriched/standard definition ALWAYS taken from the full KG (--cr_subset_dataset),
# so results are directly comparable to the previous (non-Plan-B) study.
#
# Hyperparameters (embed/layers/lr/epoch/alpha) are IDENTICAL to the previous
# study. alpha = 0.1.
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/Model"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/planB_holdout_study/logs}"

FULL_DATASET="${FULL_DATASET:-dataset-aromatique-crhkge-ready}"
HOLDOUT_DATASET="${HOLDOUT_DATASET:-dataset-aromatique-crhkge-holdout}"
# enriched/standard subsets are always defined from the FULL KG.
SUBSET_DATASET="${SUBSET_DATASET:-$FULL_DATASET}"

CR_ALPHA="${CR_ALPHA:-0.1}"

# Seeds: default 10, with a 5-seed FAST fallback. Override SEEDS directly to pick.
SEEDS_DEFAULT="2019 2020 2021 2022 2023 2024 2025 2026 2027 2028"
SEEDS_FAST="${SEEDS_FAST:-2019 2020 2021 2022 2023}"
if [ "${FAST:-0}" = "1" ]; then
  SEEDS="${SEEDS:-$SEEDS_FAST}"
else
  SEEDS="${SEEDS:-$SEEDS_DEFAULT}"
fi

mkdir -p "$LOG_DIR"

# --- Ensure both datasets exist (building data is NOT training) ---------------
if [ ! -d "$ROOT_DIR/$FULL_DATASET" ]; then
  echo "==> Building full-KG control dataset: $FULL_DATASET"
  python "$ROOT_DIR/scripts/build_cr_hkge_ready_dataset.py" --output-dataset "$FULL_DATASET"
fi
if [ ! -d "$ROOT_DIR/$HOLDOUT_DATASET" ]; then
  echo "==> Building hold-out dataset: $HOLDOUT_DATASET"
  python "$ROOT_DIR/scripts/build_cr_hkge_ready_dataset.py" \
    --holdout_mode --output-dataset "$HOLDOUT_DATASET"
fi

COMMON_ARGS=(
  --data_path ../
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
  --epoch "${CR_HKGE_PLANB_EPOCHS:-100}"
  --save_flag 1
)

# Full CR-HKGE Plan B stem: identical to the previous study's stem + Plan B gate ON.
cr_planB_stem() {
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
--cr_planB_gate 1 \
--cr_best_metric ndcg --cr_best_k 3 \
--cr_model_version cr_hkge_planB"
}

# cell -> "<dataset> :: <model flags>"
cell_spec() {
  case "$1" in
    kgat_fullkg)     echo "$FULL_DATASET :: --model_type kgat" ;;
    crhkgeB_fullkg)  echo "$FULL_DATASET :: $(cr_planB_stem)" ;;
    kgat_holdout)    echo "$HOLDOUT_DATASET :: --model_type kgat" ;;
    crhkgeB_holdout) echo "$HOLDOUT_DATASET :: $(cr_planB_stem)" ;;
    *) echo "unknown cell: $1" >&2; return 1 ;;
  esac
}

run_cell() {
  local cell="$1"
  local spec dataset flags
  spec="$(cell_spec "$cell")"
  dataset="${spec%% :: *}"
  flags="${spec#* :: }"
  read -r -a extra_args <<< "$flags"

  for seed in $SEEDS; do
    local tag="${cell}_seed${seed}"
    local train_log="$LOG_DIR/${tag}_train.log"
    local eval_log="$LOG_DIR/${tag}_subset_eval.log"
    local weights_path="../planB_holdout_study/${cell}/seed_${seed}/"

    echo "==> Training $cell on $dataset (seed=$seed)"
    (
      cd "$MODEL_DIR"
      python Main.py \
        --weights_path "$weights_path" \
        --dataset "$dataset" \
        --seed "$seed" \
        "${COMMON_ARGS[@]}" \
        "${extra_args[@]}"
    ) 2>&1 | tee "$train_log"

    echo "==> Evaluating $cell on $dataset (seed=$seed); subsets from $SUBSET_DATASET"
    (
      cd "$MODEL_DIR"
      python evaluate_item_subsets.py \
        --weights_path "$weights_path" \
        --dataset "$dataset" \
        --seed "$seed" \
        --cr_subset_data_path ../ \
        --cr_subset_dataset "$SUBSET_DATASET" \
        "${COMMON_ARGS[@]}" \
        "${extra_args[@]}"
    ) 2>&1 | tee "$eval_log"

    echo "==> Finished $cell (seed=$seed)"
  done
}

# --- Plan B invariant check (once, no training) -------------------------------
# Verifies that standard products receive EXACTLY the plain-KGAT update before we
# spend GPU hours. Runs on the full KG (which has enriched products).
if [ "${PLANB_SKIP_INVARIANT:-0}" != "1" ]; then
  echo "==> Plan B invariant check on $FULL_DATASET"
  (
    cd "$MODEL_DIR"
    python check_planb_invariant.py \
      --data_path ../ --dataset "$FULL_DATASET" \
      $(cr_planB_stem) \
      --layer_size '[64,32,16]' --embed_size 64 --kge_size 64 \
      --regs '[1e-5,1e-5]' --alg_type bi --adj_type si --adj_uni_type sum
  ) 2>&1 | tee "$LOG_DIR/planB_invariant_check.log"
fi

if [ "$#" -eq 0 ]; then
  cells=(
    kgat_fullkg
    crhkgeB_fullkg
    kgat_holdout
    crhkgeB_holdout
  )
else
  cells=("$@")
fi

for cell in "${cells[@]}"; do
  run_cell "$cell"
done

# Aggregated 2x2 table with mean +/- std across seeds, SAME CSV format as before.
python "$ROOT_DIR/scripts/summarize_multiseed_study.py" \
  --csv "$LOG_DIR/planB_holdout_summary_meanstd.csv" \
  "$LOG_DIR"/*_subset_eval.log \
  | tee "$LOG_DIR/planB_holdout_summary_meanstd.md"
