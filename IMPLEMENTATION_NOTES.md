# CR-HKGE — Implementation Notes

This document describes four changes made to break the circular evaluation in
CR-HKGE and to fix three evaluation issues. **No training was run** — all code
paths were syntax-checked and the dataset builder + summarizer were executed and
verified on the real data. Training is expected to be run on Colab.

Summary of deliverables:

| # | Deliverable | File(s) |
|---|---|---|
| 1 | Hold-out dataset build mode (+ control) | `scripts/build_cr_hkge_ready_dataset.py` |
| 2 | CFKG optimizer fix (Adam) | `Model/CFKG.py`, `Model/utility/parser.py` |
| 3 | Fair, single-flag ablations + consistent alpha | `scripts/run_cr_hkge_final_study.sh` |
| 4 | Multi-seed runs + mean±std aggregation | `scripts/run_cr_hkge_final_study.sh`, `scripts/summarize_multiseed_study.py`, `Model/Main.py`, `Model/evaluate_item_subsets.py`, `Model/utility/parser.py` |
| 5 | 2×2 hold-out study driver | `scripts/run_cr_hkge_holdout_study.sh` |
| 6 | This file | `IMPLEMENTATION_NOTES.md` |

---

## TASK 1 — Hold-out protocol (circularity-breaking)

### The problem
The positive-pair labels (`train.txt` / `test.txt`) are scored from KG relations
(`scripts/build_cr_hkge_ready_dataset.py`, `SCORING_WEIGHTS`). Three relations
encode the **direct cross-reference path** that the labels reward, and the model
also **trains** on exactly those edges. Training and testing on the same edges is
circular and inflates CR-HKGE's apparent gain.

### What was added
A new flag `--holdout_mode` on `scripts/build_cr_hkge_ready_dataset.py`.

- The **labels are computed exactly as before**, from the *full* source KG, and
  written first. They are therefore **byte-identical** to the standard run
  (verified: `diff` of `train.txt` and `test.txt` between the full and hold-out
  datasets reports IDENTICAL).
- **Only the training graph** `kg_final.txt` is then pruned: the direct
  cross-reference edges are deleted, the indirect content edges are kept.

### Exactly which edges are removed (verified counts)

| Relation | ID | Role | Hold-out | Edges |
|---|---|---|---|---|
| `inspired_by` | 0 | product → global reference (direct) | **REMOVED** | 243 |
| `has_global_accord` | 5 | global ref → global accord (direct) | **REMOVED** | 2017 |
| `belongs_to_global_family` | 6 | global ref → global family (direct) | **REMOVED** | 231 |
| `has_accord` | 1 | product → accord (indirect content) | kept | 1629 |
| `has_visual_note` | 2 | product → visual note (indirect content) | kept | 680 |
| `belongs_to_family` | 3 | product → local family (indirect content) | kept | 340 |
| `sem_similar` | 4 | product ↔ product (indirect content) | kept | 4110 |

**Total edges removed: 2491. Training-graph edges kept: 6759 (of 9250).**
These counts are written to `summary.json` under the `holdout` key and printed at
build time.

### The control variant
Running the builder **without** `--holdout_mode` produces the full-KG dataset
(`dataset-aromatique-crhkge-ready`) — this *is* the control. The two are built by
the same code path from the same labels, so "full-KG" vs "hold-out" differ **only**
in which training edges are visible.

### Build commands
```bash
# Control (full KG) — labels + complete kg_final.txt
python scripts/build_cr_hkge_ready_dataset.py \
  --output-dataset dataset-aromatique-crhkge-ready --overwrite

# Hold-out KG — same labels, direct edges removed from kg_final.txt
python scripts/build_cr_hkge_ready_dataset.py \
  --holdout_mode --output-dataset dataset-aromatique-crhkge-holdout --overwrite
```
If `--output-dataset` is omitted, the default is
`dataset-aromatique-crhkge-ready` normally and
`dataset-aromatique-crhkge-holdout` when `--holdout_mode` is set.

### Important: subset definition under hold-out
`evaluate_item_subsets.py` defines the **enriched** subset as "products with an
`inspired_by` edge in `kg_final.txt`". The hold-out `kg_final.txt` has **no**
`inspired_by` edges, so enriched/standard must be derived from the **full** KG.
The 2×2 driver does this automatically by passing
`--cr_subset_data_path ../ --cr_subset_dataset dataset-aromatique-crhkge-ready`,
keeping the enriched/standard definition identical for all four cells.

---

## TASK 2 — CFKG baseline optimizer

`Model/CFKG.py` hard-coded `tf.train.GradientDescentOptimizer` (plain SGD) while
**every** other model (`BPRMF`, `CKE`, `NFM`, `KGAT`, `CR-HKGE`) uses
`tf.train.AdamOptimizer`. At `lr=1e-4` plain SGD barely moves the TransE
embeddings, so CFKG's near-zero result was an optimizer artifact, not a model
property.

### Change
- New flag `--cfkg_optimizer` (`Model/utility/parser.py`), **default `adam`**.
- `Model/CFKG.py` now reads `args.cfkg_optimizer` and builds Adam by default; pass
  `--cfkg_optimizer sgd` to reproduce the legacy behaviour.
- Nothing else about CFKG changed (margin, L1/L2, loss, inference untouched).

---

## TASK 3 — Ablation confounds removed

### The problems in the original `run_cr_hkge_final_study.sh`
- The headline `cr_hkge_final` ran at `--cr_cross_ref_alpha 0.5`, although the
  paper's stated final model uses **0.1**.
- `A_no_relation_attention` changed **two** things at once: it set
  `cr_use_relation_weight 0` **and** `cr_relation_aware_message 0`.
- `A_no_relation_message` and `A_no_fragrance_prior` ran at `alpha 0.5`, so they
  weren't comparable to a `0.1` full model.

### The fix
All CR-HKGE variants are now built from **one canonical flag stem**
(`cr_full_stem` in the script) with **exactly one flag appended as an override**
per ablation (argparse uses the last value — verified). This makes the single
changed component literally visible in the code and guarantees consistency.

- **Alpha is `0.1` for every variant** (configurable via `CR_ALPHA=...`).
- **`cr_relation_aware_message` is held at `1` everywhere** except the one
  ablation that targets it.

### Exactly which flag differs per variant (vs `cr_hkge_final`)

| Variant | Flag changed | From → To | # flags changed |
|---|---|---|---|
| `cr_hkge_final` (full) | — | — | 0 (reference) |
| `A_no_cross_reference` | `cr_use_cross_ref` | `1 → 0` | 1 |
| `A_no_relation_attention` | `cr_use_relation_weight` | `1 → 0` | 1 |
| `A_no_relation_message` | `cr_relation_aware_message` | `1 → 0` | 1 |
| `A_no_fragrance_prior` | `cr_relation_prior_mode` | `fragrance → none` | 1 |
| `A_no_novelty_modules` (control) | `cr_use_relation_weight`, `cr_use_cross_ref`, `cr_relation_aware_message` | all `1 → 0` | 3 (by design) |

`A_no_novelty_modules` is intentionally a multi-flag control (≈ KGAT through the
CR-HKGE code path), not a single-component ablation.

Note on `A_no_relation_attention`: `cr_relation_aware_message` is kept at `1` for
consistency, but it is a *sub-feature* of relation attention — the model
(`Model/CRHKGE.py:_build_relation_aware_A_fold_hat`) returns no relation-specific
messages when `cr_use_relation_weight=0`. So holding the flag constant is correct
and the effective single change is "relation attention off".

The canonical full-model flag set (the stem):
```
--model_type cr_hkge
--cr_use_relation_weight 1   --cr_use_cross_ref 1
--cr_relation_weight_mode semantic
--cr_relation_prior_mode fragrance   --cr_relation_prior_strength 1.0
--cr_relation_attention_scale type_count
--cr_relation_aware_message 1   --cr_relation_message_scale type_count
--cr_cross_ref_bi_interaction 0   --cr_cross_ref_gate 0
--cr_cross_ref_alpha 0.1
--cr_best_metric ndcg   --cr_best_k 3
```

---

## TASK 4 — Multi-seed runs + variance

### Changes
- New `--seed` flag (`Model/utility/parser.py`, default `2019`).
- `Model/Main.py` and `Model/evaluate_item_subsets.py` now seed `tf` / `numpy` /
  `random` from `args.seed` (was hard-coded `2019`); seeding moved to *after*
  `parse_args()`.
- Both run scripts loop over `SEEDS` (default `2019 2020 2021 2022 2023`,
  configurable via the `SEEDS` env var). Each seed trains into its **own** weights
  directory (`.../seed_<seed>/`) so checkpoints never collide, and writes a log
  named `<target>_seed<seed>_subset_eval.log`.
- New aggregator `scripts/summarize_multiseed_study.py` groups eval logs by target
  (ignoring the seed), and reports **mean ± sample std** for every scope / metric /
  K, plus an optional tidy CSV (`--csv`). Verified on synthetic logs.

`run_cr_hkge_final_study.sh` now emits both:
- `final_summary_per_seed.md` — one row per (target, seed),
- `final_summary_meanstd.md` + `final_summary_meanstd.csv` — the aggregated table.

---

## DELIVERABLE 4 — 2×2 hold-out study driver

`scripts/run_cr_hkge_holdout_study.sh` trains, **on the same seeds and the same
labels**, the four cells:

|            | full KG (control)  | hold-out KG          |
|------------|--------------------|----------------------|
| **KGAT**   | `kgat_fullkg`      | `kgat_holdout`       |
| **CR-HKGE**| `crhkge_fullkg`    | `crhkge_holdout`     |

- CR-HKGE cells use the same `alpha=0.1` stem as the headline model.
- The script auto-builds the full and hold-out datasets if missing (building data
  is **not** training).
- enriched/standard subsets are defined from the **full** KG for all four cells
  (see Task 1 note), so the only thing that changes between the full and hold-out
  columns is which training edges the model saw.
- Output: `holdout_summary_meanstd.md` + `holdout_summary_meanstd.csv`.

The headline read of the 2×2: compare the **full → hold-out drop** for KGAT vs for
CR-HKGE. A CR-HKGE advantage that **survives** hold-out is real cross-reference
generalization; an advantage that **collapses** to ≈ KGAT under hold-out was
driven by training on the same edges the labels were built from.

---

## How to run on Colab

```bash
# 0) From the repo root, after restoring Model/weights as usual.
#    (Building datasets is cheap and not training.)

# 1) Build the hold-out dataset (control already exists as the default build).
python scripts/build_cr_hkge_ready_dataset.py \
  --holdout_mode --output-dataset dataset-aromatique-crhkge-holdout --overwrite

# 2) Fair, multi-seed final study (baselines + CR-HKGE + single-flag ablations).
#    Configure seeds/epochs/alpha via env vars if desired.
SEEDS="2019 2020 2021 2022 2023" \
CR_HKGE_FINAL_EPOCHS=100 \
CR_ALPHA=0.1 \
bash scripts/run_cr_hkge_final_study.sh

# 3) The 2x2 circularity-breaking study (KGAT/CR-HKGE × full/hold-out).
SEEDS="2019 2020 2021 2022 2023" \
CR_HKGE_HOLDOUT_EPOCHS=100 \
bash scripts/run_cr_hkge_holdout_study.sh

# Quick smoke test (one seed, few epochs) before a full run:
SEEDS="2019" CR_HKGE_FINAL_EPOCHS=2 bash scripts/run_cr_hkge_final_study.sh cr_hkge_final
```

Useful env vars: `SEEDS`, `CR_ALPHA`, `DATASET`, `FULL_DATASET`,
`HOLDOUT_DATASET`, `SUBSET_DATASET`, `CR_HKGE_FINAL_EPOCHS`,
`CR_HKGE_HOLDOUT_EPOCHS`, `LOG_DIR`. Both run scripts also accept an explicit list
of targets/cells as positional args (e.g. `bash scripts/run_cr_hkge_holdout_study.sh kgat_holdout crhkge_holdout`).

---

## Output files — what each means

### Dataset folders (`dataset-aromatique-crhkge-ready`, `…-holdout`)
| File | Meaning |
|---|---|
| `train.txt` / `test.txt` | Positive pairs (labels). **Identical** between full and hold-out. |
| `kg_final.txt` | Training graph. Full = all 9250 edges; hold-out = 6759 edges (direct cross-reference edges removed). |
| `summary.json` | Dataset stats incl. `holdout` block (removed relations + counts). |
| `positive_pair_scores.tsv` | Per-pair score audit. |
| `profile2product.tsv` | profile → source product mapping. |
| `README.md` | Auto-generated; hold-out build adds a "HOLD-OUT MODE" section. |

### Study logs (`final_study/logs/`, `holdout_study/logs/`)
| File | Meaning |
|---|---|
| `<target>_seed<seed>_train.log` | Training stdout for one (target, seed). |
| `<target>_seed<seed>_subset_eval.log` | overall / enriched / standard Top-K metrics for one (target, seed). |
| `final_summary_per_seed.md` | One row per (target, seed). |
| `final_summary_meanstd.md` / `.csv` | **Mean ± std across seeds** per target — the headline table for significance. |
| `holdout_summary_meanstd.md` / `.csv` | The aggregated 2×2 (full vs hold-out) table. |

### Per-seed weights
`final_study/<target>/seed_<seed>/weights/...` and
`holdout_study/<cell>/seed_<seed>/weights/...` — isolated checkpoints so seeds and
cells never overwrite each other; the evaluator restores from the matching path.
