#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CR-HKGE Plan C+ (residual enrichment + discriminative contrastive) 2x2 study
#
# Plan C+ leaves KGAT message passing UNTOUCHED (no lambda_r, no relation-aware
# message, no in-graph attention edits) and adds:
#   - component 1: a learnable residual at the FINAL embedding, ONLY for enriched
#                  products (--cr_use_residual, gamma init 0.1);
#   - component 2: a DISCRIMINATIVE contrastive loss that pushes apart
#                  attribute-similar but reference-different products
#                  (--cr_use_contrastive, lambda_c 0.1). ANTI-CIRCULAR: it never
#                  attracts same_global_reference pairs (that is the test label).
# Both components are INDEPENDENTLY toggleable via env vars below.
#
#                        | full KG (control)     | hold-out KG (direct paths removed)
#   ---------------------+-----------------------+-----------------------------------
#   KGAT                 | kgat_fullkg           | kgat_holdout
#   CR-HKGE-C+           | crhkgeCplus_fullkg    | crhkgeCplus_holdout
#
# Each cell is evaluated on overall/enriched/standard (subsets from the FULL KG).
# KEY question: does C+'s gain SURVIVE hold-out (i.e. is the new signal
# non-circular)? On the hold-out KG inspired_by/global edges are gone, so the
# residual context is empty and there are no contrastive anchors -> C+ provably
# reduces to KGAT (a safe floor, unlike Plan B which underperformed KGAT).
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/Model"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/planCplus_holdout_study/logs}"

FULL_DATASET="${FULL_DATASET:-dataset-aromatique-crhkge-ready}"
HOLDOUT_DATASET="${HOLDOUT_DATASET:-dataset-aromatique-crhkge-holdout}"
SUBSET_DATASET="${SUBSET_DATASET:-$FULL_DATASET}"

# Independent component toggles (Task 3): residual-only -> CR_USE_CONTRASTIVE=0,
# contrastive-only -> CR_USE_RESIDUAL=0, both (default) -> leave unset.
CR_USE_RESIDUAL="${CR_USE_RESIDUAL:-1}"
CR_RESIDUAL_GAMMA="${CR_RESIDUAL_GAMMA:-0.1}"
CR_USE_CONTRASTIVE="${CR_USE_CONTRASTIVE:-1}"
CR_CONTRASTIVE_WEIGHT="${CR_CONTRASTIVE_WEIGHT:-0.1}"
CR_CONTRASTIVE_MARGIN="${CR_CONTRASTIVE_MARGIN:-1.0}"
CR_CONTRASTIVE_NEGS="${CR_CONTRASTIVE_NEGS:-5}"

# Seeds: QUICK (3) default; FULL=1 -> 5 seeds. Or set SEEDS directly.
SEEDS_QUICK="2019 2020 2021"
SEEDS_FULL="${SEEDS_FULL:-2019 2020 2021 2022 2023}"
if [ "${FULL:-0}" = "1" ]; then
  SEEDS="${SEEDS:-$SEEDS_FULL}"
else
  SEEDS="${SEEDS:-$SEEDS_QUICK}"
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
  --epoch "${CR_HKGE_CPLUS_EPOCHS:-100}"
  --save_flag 1
)

# CR-HKGE-C+ stem: KGAT message passing OFF (pure KGAT) + residual + contrastive.
cr_cplus_stem() {
  echo "--model_type cr_hkge \
--cr_use_relation_weight 0 \
--cr_use_cross_ref 0 \
--cr_relation_aware_message 0 \
--cr_use_residual ${CR_USE_RESIDUAL} \
--cr_residual_gamma ${CR_RESIDUAL_GAMMA} \
--cr_use_contrastive ${CR_USE_CONTRASTIVE} \
--cr_contrastive_weight ${CR_CONTRASTIVE_WEIGHT} \
--cr_contrastive_margin ${CR_CONTRASTIVE_MARGIN} \
--cr_contrastive_negs ${CR_CONTRASTIVE_NEGS} \
--cr_best_metric ndcg --cr_best_k 3 \
--cr_model_version cr_hkge_planCplus"
}

cell_spec() {
  case "$1" in
    kgat_fullkg)         echo "$FULL_DATASET :: --model_type kgat" ;;
    crhkgeCplus_fullkg)  echo "$FULL_DATASET :: $(cr_cplus_stem)" ;;
    kgat_holdout)        echo "$HOLDOUT_DATASET :: --model_type kgat" ;;
    crhkgeCplus_holdout) echo "$HOLDOUT_DATASET :: $(cr_cplus_stem)" ;;
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
    local weights_path="../planCplus_holdout_study/${cell}/seed_${seed}/"

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

# --- Plan C+ invariant check (once, no training) ------------------------------
if [ "${CPLUS_SKIP_INVARIANT:-0}" != "1" ]; then
  echo "==> Plan C+ invariant check on $FULL_DATASET"
  (
    cd "$MODEL_DIR"
    python check_planCplus_invariant.py \
      --data_path ../ --dataset "$FULL_DATASET" \
      $(cr_cplus_stem) \
      --layer_size '[64,32,16]' --embed_size 64 --kge_size 64 \
      --regs '[1e-5,1e-5]' --alg_type bi --adj_type si --adj_uni_type sum
  ) 2>&1 | tee "$LOG_DIR/planCplus_invariant_check.log"
fi

if [ "$#" -eq 0 ]; then
  cells=(
    kgat_fullkg
    crhkgeCplus_fullkg
    kgat_holdout
    crhkgeCplus_holdout
  )
else
  cells=("$@")
fi

for cell in "${cells[@]}"; do
  run_cell "$cell"
done

# Aggregated 2x2 table, SAME CSV format as holdout_summary_meanstd.csv.
python "$ROOT_DIR/scripts/summarize_multiseed_study.py" \
  --csv "$LOG_DIR/planCplus_holdout_summary_meanstd.csv" \
  "$LOG_DIR"/*_subset_eval.log \
  | tee "$LOG_DIR/planCplus_holdout_summary_meanstd.md"
