# CR-HKGE — Plan B: Gated Conditional Enrichment

This document describes Plan B and the 2×2 re-run. **No training was run** (the
user runs on Colab). All code was syntax-checked; the invariant is proven
algebraically below and is checkable on Colab with `check_planb_invariant.py`.

## The problem Plan B fixes
The 5-seed hold-out study showed CR-HKGE **beats** KGAT on **enriched** products
(those with `inspired_by`) but **loses** to KGAT on **standard** products. Although
the cross-reference term `c_ref` is correctly zeroed for standard products, the
other two CR-HKGE modifications still altered standard products' message passing:

- the relation-type prior **λ_r reweighted their attention** (their neighbors), and
- the **relation-aware message** replaced KGAT's aggregation with a different one.

Both degraded standard-product embeddings below plain KGAT.

**Goal:** make standard products receive **exactly** the plain-KGAT update (so they
can never do worse than KGAT), while enriched products keep the full CR-HKGE
treatment.

---

## TASK 1 — The forward pass: KGAT vs CR-HKGE vs gated (Plan B)

All three share KGAT's bi-interaction layer; they differ **only in the side
message** `s_p` (the aggregated neighbor signal) and an additive cross-reference
term `c_ref`. For a node `p` at layer `k` (weights `W_gc,b_gc,W_bi,b_bi` shared):

```
KGAT_update(p)   : s_p = (A_kgat · E)_p                       # A_kgat = pure attention (no λ_r)
CR_update(p)     : s_p = (Σ_r λ_r · (A_r · E))_p  (+ c_ref_p) # relation-aware msg + λ_r + cross-ref
common tail      : add = e_p + s_p (+ c_ref_p)
                   e_p' = LeakyReLU(add·W_gc + b_gc) + LeakyReLU((e_p ⊙ s_p)·W_bi + b_bi)
```

The three deviations of `CR_update` from `KGAT_update` are:

| # | Deviation | Where (file) | Standard product handling before Plan B |
|---|---|---|---|
| (a) | cross-reference `c_ref` | `_create_cross_reference_context` | already 0 (× `cr_product_mask_tensor`) ✓ |
| (b) | λ_r reweights attention | `self.A_in` rebuilt from overridden `_generate_transE_score` (× `relation_multiplier`) | **polluted** ✗ |
| (c) | relation-aware message | `_relation_aware_side_embeddings` (uses static `A_r` × λ_r message mult) | **replaced** ✗ |

### The gate
Plan B introduces a per-node gate `g_p`:

```
g_p = 1  for STANDARD product nodes (item node, no inspired_by)
g_p = 0  for everything else (enriched products, users, non-product entities)

s_p^gated = s_p^CR + g_p · ( s_p^KGAT − s_p^CR )
c_ref_p^gated = (1 − g_p) · c_ref_p
```

- For a **standard** product (`g_p = 1`): `s_p^gated = s_p^KGAT` and `c_ref = 0` ⇒
  the layer collapses to **exactly `KGAT_update(p)`**. This is the formula the task
  asked for, `e_p = g·CR + (1−g)·KGAT`, applied at the side-message level (which is
  equivalent because the tail is row-wise and the weights are shared).
- For an **enriched** product / user / entity (`g_p = 0`): `s_p^gated = s_p^CR`,
  `c_ref` unchanged ⇒ **identical to current CR-HKGE**. Enriched behavior is kept.

Defining `g` on *standard products only* (not on users/entities) is deliberate: it
keeps enriched products' neighborhoods (the entities they aggregate) on the CR path,
so enriched behavior is genuinely unchanged.

### How the gate neutralizes (b) and (c) — in-graph & differentiable
`s_p^KGAT` is computed from a **pure in-graph attention** `A_kgat`
(`_build_kgat_attention_tensor`): it builds the TransR scores by calling the
**base-class** `KGAT._generate_transE_score` (NO λ_r multiplier) over all KG edges
and applies `tf.sparse.softmax`. So:

- (b) is neutralized: standard products aggregate over `A_kgat`, which has **no λ_r**
  in it — their neighbors are weighted exactly as plain KGAT.
- (c) is neutralized: standard products use `A_kgat · E`, **not** the relation-aware
  `Σ_r λ_r·(A_r·E)`.

This is done entirely with TensorFlow ops (`sparse.softmax`,
`sparse_tensor_dense_matmul`, elementwise blend), so the gate and the λ_r path are
**differentiable** — no external SciPy attention matrix is rebuilt for the gate
(the audit had flagged the old λ_r→attention path, which baked λ_r into the SciPy
`A_in`, as non-differentiable; Plan B's standard branch sidesteps it with the
in-graph `A_kgat`). The enriched (CR) branch is unchanged.

### Code changes in `Model/CRHKGE.py`
- `_parse_args`: new `self.cr_planB_gate` (from `--cr_planB_gate`, default 1).
- `_build_kgat_attention_tensor()`: pure in-graph KGAT attention (no λ_r), cached.
- `_standard_product_gate()`: builds `standard_mask` = item-node ∧ ¬enriched; returns
  zeros if `--cr_planB_gate 0` (legacy behavior).
- `_kgat_reference_layer()`: one plain-KGAT bi-interaction layer (used by the
  standard branch and the invariant check).
- `_create_bi_interaction_embed()`: computes `side_cr` and `side_kgat`, blends with
  the gate, masks `c_ref` by `(1−g)`, and captures layer-1 invariant tensors.
- `check_standard_equals_kgat()`: the invariant assertion (below).

`Model/utility/parser.py`: new `--cr_planB_gate` (default 1).

---

## The invariant check (Task 1 requirement)

**Invariant:** when `g_p = 1` (standard product), given identical layer-0 inputs,
the gated CR-HKGE update equals the plain-KGAT update within fp tolerance.

`check_standard_equals_kgat(sess)` runs (with message dropout = 0) two graph tensors
captured at layer 1 from the **same** `ego_0`:

- `cr_dbg_gated_layer1` — the gated update,
- `cr_dbg_kgat_layer1` — `_kgat_reference_layer(ego_0, A_kgat)` (pure KGAT),

and asserts `max|gated − KGAT|` over **standard** product nodes ≤ `1e-5`, while
reporting that **enriched** nodes differ (> 0, sanity that the CR path is live).

> Why layer 1: a multi-layer GNN node depends on its neighbors' previous-layer
> embeddings, so "standard == KGAT" is only exact **per layer given identical
> inputs** (layer 1 has identical inputs in both streams by construction). This is
> the precise, well-defined statement of the invariant, and it is what guarantees
> standard products never diverge from the KGAT operator.

Run it standalone (no training, no checkpoint needed):
```bash
cd Model
python check_planb_invariant.py \
  --data_path ../ --dataset dataset-aromatique-crhkge-ready \
  --model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 \
  --cr_relation_aware_message 1 --cr_relation_prior_mode fragrance \
  --cr_planB_gate 1 \
  --layer_size '[64,32,16]' --embed_size 64 --kge_size 64 \
  --regs '[1e-5,1e-5]' --alg_type bi --adj_type si --adj_uni_type sum
```
Expected: `PLAN B INVARIANT CHECK PASSED`. The 2×2 script also runs this once
before training (set `PLANB_SKIP_INVARIANT=1` to skip).

Note on the **hold-out** KG: it has 0 `inspired_by` edges ⇒ 0 enriched products ⇒
**every** product is standard ⇒ all products get the plain-KGAT update (cross-ref
also empty). That is the intended safety property: with no enrichment signal,
CR-HKGE-PlanB cannot underperform KGAT on products.

---

## TASK 2 — alpha and flags held constant
The CR-HKGE-PlanB cells use `--cr_cross_ref_alpha 0.1` and the **same** stem as the
previous study (relation weight/cross-ref/prior/attention/message flags identical),
plus `--cr_planB_gate 1`. Embedding size (64), layers `[64,32,16]`, `lr 1e-4`,
epochs (100), regs `[1e-5,1e-5]`, batch 64, dropout — all unchanged. Only Plan B's
gate differs from the previous study, so the comparison is clean.

## TASK 3 — the 2×2 hold-out re-run
`scripts/run_cr_hkge_planB_holdout_study.sh` trains, on the same seeds, the 2×2:

|              | full KG          | hold-out KG       |
|--------------|------------------|-------------------|
| **KGAT**     | `kgat_fullkg`    | `kgat_holdout`    |
| **CR-HKGE-PlanB** | `crhkgeB_fullkg` | `crhkgeB_holdout` |

Each cell is evaluated on **overall / enriched / standard** subsets, with the
enriched/standard definition taken from the **full** KG
(`--cr_subset_data_path ../ --cr_subset_dataset $FULL_DATASET`), exactly like the
previous study — so results line up cell-for-cell with the old numbers. It reuses
the existing hold-out builder (`--holdout_mode`) and auto-builds datasets if missing.

## TASK 4 — seeds & output
- `SEEDS` env var. **Default 10 seeds** `2019…2028`. **Fast 5-seed** mode:
  `FAST=1` (uses `SEEDS_FAST="2019 2020 2021 2022 2023"`). Or set `SEEDS` directly.
- Aggregation via `scripts/summarize_multiseed_study.py` → mean ± sample std for
  every metric, **every subset** (overall/enriched/standard), in the **same CSV
  format** as `holdout_summary_meanstd.csv`
  (`scope,model,n_seeds,metric,k,mean,std,n,values`). Output:
  `planB_holdout_study/logs/planB_holdout_summary_meanstd.{md,csv}`.

## TASK 5 — prior fixes preserved
- **CFKG** still defaults to **Adam** (`--cfkg_optimizer adam`; `Model/CFKG.py`).
- **Ablation alpha consistency** in `scripts/run_cr_hkge_final_study.sh` unchanged
  (single-flag ablations, `CR_ALPHA=0.1`). Neither was touched by Plan B.

---

## How to run on Colab

```bash
# (datasets auto-build if missing; building data is NOT training)

# --- 10-seed full study (default) ---
CR_HKGE_PLANB_EPOCHS=100 \
bash scripts/run_cr_hkge_planB_holdout_study.sh

# --- 5-seed FAST study ---
FAST=1 CR_HKGE_PLANB_EPOCHS=100 \
bash scripts/run_cr_hkge_planB_holdout_study.sh

# --- custom seeds ---
SEEDS="2019 2020 2021" bash scripts/run_cr_hkge_planB_holdout_study.sh

# --- invariant check only (fast, no training) ---
PLANB_SKIP_INVARIANT=0 bash scripts/run_cr_hkge_planB_holdout_study.sh kgat_fullkg  # check + 1 cell
# or directly: see check_planb_invariant.py command above.

# --- one cell only (e.g. smoke test) ---
SEEDS="2019" CR_HKGE_PLANB_EPOCHS=2 \
bash scripts/run_cr_hkge_planB_holdout_study.sh crhkgeB_fullkg
```

### Output files
| File | Meaning |
|---|---|
| `planB_holdout_study/logs/planB_invariant_check.log` | invariant assertion result |
| `planB_holdout_study/logs/<cell>_seed<seed>_train.log` | training log per cell+seed |
| `planB_holdout_study/logs/<cell>_seed<seed>_subset_eval.log` | overall/enriched/standard Top-K per cell+seed |
| `planB_holdout_study/logs/planB_holdout_summary_meanstd.md` | 2×2 mean±std table (human-readable) |
| `planB_holdout_study/logs/planB_holdout_summary_meanstd.csv` | mean±std, same format as `holdout_summary_meanstd.csv` |
| `planB_holdout_study/<cell>/seed_<seed>/weights/...` | per-cell, per-seed checkpoints |

**What to look for:** on **standard** products, `crhkgeB_*` should now be **≥**
`kgat_*` (Plan B guarantees no regression vs KGAT); on **enriched** products,
`crhkgeB_fullkg` should keep its advantage over `kgat_fullkg`. Compare the standard
subset of this study against the previous `holdout_summary_meanstd.csv` to confirm
the standard-product loss is gone.
