#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# CR-HKGE hold-out 2x2 study (circularity-breaking evaluation)
#
# Trains FOUR cells on the SAME seeds and SAME labels:
#
#                     | full KG (control)     | hold-out KG (direct paths removed)
#   ------------------+-----------------------+-----------------------------------
#   KGAT              | kgat_fullkg           | kgat_holdout
#   CR-HKGE (alpha)   | crhkge_fullkg         | crhkge_holdout
#
# The hold-out KG has the DIRECT cross-reference edges removed from the TRAINING
# graph (inspired_by, has_global_accord, belongs_to_global_family) while the
# labels (train.txt/test.txt) are byte-identical to the full-KG dataset. So the
# gap (full -> holdout) quantifies how much of each model's score depends on
# literally training on the same edges the labels were built from.
#
# enriched/standard subsets are ALWAYS defined from the FULL KG (via
# --cr_subset_dataset), because the hold-out kg_final.txt has no inspired_by
# edges to define them from. This keeps the subset definition identical across
# all four cells.
# =============================================================================

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT_DIR/Model"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/holdout_study/logs}"

FULL_DATASET="${FULL_DATASET:-dataset-aromatique-crhkge-ready}"
HOLDOUT_DATASET="${HOLDOUT_DATASET:-dataset-aromatique-crhkge-holdout}"
# Subset (enriched/standard) definition source: always the FULL KG.
SUBSET_DATASET="${SUBSET_DATASET:-$FULL_DATASET}"

CR_ALPHA="${CR_ALPHA:-0.1}"
SEEDS="${SEEDS:-2019 2020 2021 2022 2023}"

mkdir -p "$LOG_DIR"

# --- Ensure both datasets exist (building data is NOT training) ---------------
if [ ! -d "$ROOT_DIR/$FULL_DATASET" ]; then
  echo "==> Building full-KG control dataset: $FULL_DATASET"
  python "$ROOT_DIR/scripts/build_cr_hkge_ready_dataset.py" \
    --output-dataset "$FULL_DATASET"
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
  --epoch "${CR_HKGE_HOLDOUT_EPOCHS:-100}"
  --save_flag 1
)

# Full CR-HKGE flag stem (alpha = $CR_ALPHA, 0.1 by default), identical to the
# headline model in run_cr_hkge_final_study.sh.
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

# cell -> "<dataset> :: <model flags>"
cell_spec() {
  case "$1" in
    kgat_fullkg)    echo "$FULL_DATASET :: --model_type kgat" ;;
    crhkge_fullkg)  echo "$FULL_DATASET :: $(cr_full_stem)" ;;
    kgat_holdout)   echo "$HOLDOUT_DATASET :: --model_type kgat" ;;
    crhkge_holdout) echo "$HOLDOUT_DATASET :: $(cr_full_stem)" ;;
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
    local weights_path="../holdout_study/${cell}/seed_${seed}/"

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

if [ "$#" -eq 0 ]; then
  cells=(
    kgat_fullkg
    crhkge_fullkg
    kgat_holdout
    crhkge_holdout
  )
else
  cells=("$@")
fi

for cell in "${cells[@]}"; do
  run_cell "$cell"
done

# Aggregated 2x2 table with mean +/- std across seeds.
python "$ROOT_DIR/scripts/summarize_multiseed_study.py" \
  --csv "$LOG_DIR/holdout_summary_meanstd.csv" \
  "$LOG_DIR"/*_subset_eval.log \
  | tee "$LOG_DIR/holdout_summary_meanstd.md"
