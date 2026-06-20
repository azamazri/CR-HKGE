# CR-HKGE — Plan C+: Residual Enrichment + Discriminative Contrastive Loss

**No training was run** (the user runs on Colab, ~9 min/run). Code is
syntax-checked; the negative-sampling design is validated on the real KG (below);
the invariants are proven and runtime-checkable with `check_planCplus_invariant.py`.

## Why Plan C+ (vs B)
Plan B touched message passing, which couples enriched↔standard (1066 `sem_similar`
edges) and dragged enriched products down. Plan C+ therefore:
1. leaves the **entire KGAT forward pass identical** to original KGAT (no λ_r, no
   relation-aware message, no in-graph attention edits);
2. adds a **residual at the final embedding**, only for enriched products;
3. adds a **discriminative contrastive loss** that injects new signal.
Both components are **independently toggleable**.

---

## Component 1 — Residual enrichment (Task 1)

KGAT produces, per product `p`, the final layer-aggregated embedding `e_p_kgat`
(the concat of all layers, dim `64+64+32+16 = 176`). Plan C+ adds:

```
e_p_final = e_p_kgat + g_p · γ · c_ref(p)
```

- `g_p = cr_product_mask` (1 enriched / 0 standard) — applied as a mask inside `c_ref`.
- `γ` = **learnable scalar**, init `0.1` (`--cr_residual_gamma`), variable `cr_residual_gamma`.
- `c_ref(p)` = cross-reference context (global-attribute attention → global
  reference → product via `inspired_by`), computed **once at the final embedding**
  by `_create_residual_cross_reference_context`, which reproduces the propagation of
  `_create_cross_reference_context` at the final (176-d) dimension with its own
  projection `W_cr_residual`/`b_cr_residual` (no λ_r/α/gate confounds; `γ` scales it).
- Toggle: `--cr_use_residual` (default 1).

**Where it lives:** `CRHKGE._create_bi_interaction_embed` now calls the unchanged
KGAT core (`_bi_interaction_core` → `super()._create_bi_interaction_embed()`), captures
the pure-KGAT embeddings, then `_apply_residual` adds the term. KGAT message passing
is never edited.

---

## Component 2 — Discriminative contrastive loss (Task 2, ANTI-CIRCULAR)

```
L_total = L_KGAT_BPR + L_KG + λ_c · L_discriminative
L_discriminative = mean over (a, n) pairs of  relu( margin − ‖e_a − e_n‖₂ )
```

- `λ_c` = `--cr_contrastive_weight` (default 0.1); `margin` = `--cr_contrastive_margin`
  (default 1.0); toggle `--cr_use_contrastive` (default 1).
- Embeddings `e_a, e_n` are the **final** product embeddings `self.ea_embeddings`
  (residual included), gathered at precomputed constant index arrays (no per-batch
  sampling, no `Main.py` changes). The term is added to `self.loss` and the phase-I
  optimizer is rebuilt over the combined loss.

### Hard-negative sampling (`_build_contrastive_pairs`)
For each **enriched anchor** `a`, the hard negatives are the `K=--cr_contrastive_negs`
(default 5) products `n` that are **most attribute-similar** to `a`
(largest Jaccard over `has_accord ∪ belongs_to_family`) **BUT have a different
global reference** (`inspired_by` target set disjoint from `a`'s). The loss
**pushes `a` and `n` apart**.

Validated on the full KG (`dataset-aromatique-crhkge-ready`):
- 243 enriched anchors → **1215** (anchor, negative) pairs (K=5),
- **0 same-reference violations** (no negative shares the anchor's reference),
- negatives: 958 enriched-with-different-reference + 257 standard.

### Why this is NOT circular
The **test label** is *"`same_global_reference` pairs are relevant"* — an
**attractive** signal on same-reference items. The contrastive loss **never** adds a
same-reference positive (that would copy the label). It only adds the **opposite,
orthogonal** signal: *"attribute-similar but reference-different ⇒ push apart."*

- It cannot be used to reconstruct the test label, because it contains **no**
  "these two are co-relevant" information — only "these two, despite looking alike on
  attributes, are NOT reference-equivalent."
- The signal is **new** relative to attributes (which KGAT already encodes via
  `has_accord`/`sem_similar`) — it teaches the distinction *attributes alone cannot
  express*: two perfumes can share accords yet belong to different references.
- For InfoNCE stability the spec allowed a self-positive; the margin-hinge form needs
  **no positive at all**, so we use none — the BPR + KG losses supply the attractive
  structure and prevent collapse, while this term only enforces a minimum separation
  on the specific hard-negative pairs.

---

## Independent toggles (Task 3)
`--cr_use_residual` and `--cr_use_contrastive` are independent:
- residual-only: `--cr_use_residual 1 --cr_use_contrastive 0`
- contrastive-only: `--cr_use_residual 0 --cr_use_contrastive 1`
- both (default): `1 1`
The run script exposes `CR_USE_RESIDUAL` / `CR_USE_CONTRASTIVE` env vars.
**Safety net:** `_apply_residual` and the contrastive build are each wrapped in
`try/except`; a build-time failure in one prints a warning and falls back (residual →
identity, contrastive → 0) so the other still runs.

---

## The three invariants (a)(b)(c)
Checked by `CRHKGE.check_planCplus_invariants` and `Model/check_planCplus_invariant.py`
on a synthetic forward (random weights, dropout = 0):

| # | Statement | How it holds | Expected |
|---|---|---|---|
| (a) | standard products: `e_final == e_kgat` exactly | `c_ref` is masked by `g_p` (0 for standard) | **PASS** (`~1e-7`) |
| (b) | `γ=0` ∧ contrastive off ⇒ `e_final == e_kgat` for **all** products (provable KGAT floor) | residual `= γ·c_ref = 0`; loss `= L_KGAT` | **PASS** at `γ=0` |
| (c) | KGAT propagation byte-identical to original KGAT | message passing is `super()._create_bi_interaction_embed()`; with `cr_use_relation_weight 0` the attentive `A_in` is also pure (no λ_r) | **PASS** (`0`) |

Run:
```bash
cd Model
# (a) + (c), with residual active:
python check_planCplus_invariant.py --data_path ../ --dataset dataset-aromatique-crhkge-ready \
  --model_type cr_hkge --cr_use_relation_weight 0 --cr_use_cross_ref 0 --cr_relation_aware_message 0 \
  --cr_use_residual 1 --cr_residual_gamma 0.1 --cr_use_contrastive 1 \
  --layer_size '[64,32,16]' --embed_size 64 --kge_size 64 --regs '[1e-5,1e-5]' \
  --alg_type bi --adj_type si --adj_uni_type sum
# (b) floor: add  --cr_residual_gamma 0 --cr_use_contrastive 0   (then ALL products == KGAT)
```
The 2×2 script runs (a)+(c) automatically before training (`CPLUS_SKIP_INVARIANT=1` to skip).

---

## 2×2 hold-out study (Task 4) & seeds (Task 5)
`scripts/run_cr_hkge_planCplus_holdout_study.sh` trains, on the same seeds:

|            | full KG               | hold-out KG           |
|------------|-----------------------|-----------------------|
| **KGAT**   | `kgat_fullkg`         | `kgat_holdout`        |
| **CR-HKGE-C+** | `crhkgeCplus_fullkg` | `crhkgeCplus_holdout` |

- Reuses `--holdout_mode` (removes `inspired_by`/global edges from the **training**
  graph; test labels unchanged). enriched/standard subsets are taken from the **full**
  KG (`--cr_subset_dataset`), like the previous studies, so results line up cell-for-cell.
- Output: `planCplus_holdout_study/logs/planCplus_holdout_summary_meanstd.{md,csv}`,
  **same CSV format** as `holdout_summary_meanstd.csv`
  (`scope,model,n_seeds,metric,k,mean,std,n,values`).
- **Seeds:** `SEEDS_QUICK="2019 2020 2021"` (default, ~12 runs ≈ 1.8 h). `FULL=1` →
  `SEEDS_FULL="2019 2020 2021 2022 2023"`. Or set `SEEDS` directly.

### What the hold-out answers (and an honest caveat)
On the **hold-out** KG, `inspired_by`/global edges are gone, so:
- the residual context `c_ref` is **empty** → residual = 0, and
- there are **no enriched anchors** → contrastive = 0.

So **CR-HKGE-C+ provably reduces to KGAT on the hold-out KG** — a *safe floor* (it
cannot underperform KGAT, unlike Plan B). The hold-out 2×2 therefore answers:
1. **does C+ beat KGAT on the full KG?** (the residual + discriminative signal), and
2. **does C+ avoid harm when the edges are hidden?** (yes, by construction it = KGAT).

Because both components are derived from the training graph's `inspired_by`/global
edges, the *full-KG gain* is edge-dependent and does **not** transfer to the hold-out
graph (where those edges are absent). The **non-circularity** of the contrastive
signal is established by its **direction** (discriminate, never attract same-reference)
— not by surviving the hold-out, which removes the very edges the signal is defined
from. Compare `crhkgeCplus_fullkg` vs `kgat_fullkg` for the gain, and confirm
`crhkgeCplus_holdout ≈ kgat_holdout` for the floor.

---

## Task 6 — prior fixes preserved
- **CFKG** still defaults to **Adam** (`Model/CFKG.py`, `--cfkg_optimizer adam`).
- Ablation alpha consistency, hold-out builder, multi-seed summarizer — untouched.

### ⚠️ Default-behavior note
`--cr_use_residual` and `--cr_use_contrastive` default to **1** (per spec). This means
**any** `cr_hkge` run now adds the residual + contrastive **unless disabled**. To
reproduce the earlier (pre-C+) Plan B / final-study CR-HKGE results exactly, pass
`--cr_use_residual 0 --cr_use_contrastive 0`. Non-`cr_hkge` baselines
(BPRMF/CKE/NFM/CFKG/KGAT) are unaffected.

---

## How to run on Colab
```bash
# Quick (3 seeds, default) — recommended first:
CR_HKGE_CPLUS_EPOCHS=100 bash scripts/run_cr_hkge_planCplus_holdout_study.sh

# Full (5 seeds):
FULL=1 CR_HKGE_CPLUS_EPOCHS=100 bash scripts/run_cr_hkge_planCplus_holdout_study.sh

# Ablate components (independent toggles):
CR_USE_CONTRASTIVE=0 bash scripts/run_cr_hkge_planCplus_holdout_study.sh   # residual-only
CR_USE_RESIDUAL=0    bash scripts/run_cr_hkge_planCplus_holdout_study.sh   # contrastive-only

# One cell smoke test:
SEEDS="2019" CR_HKGE_CPLUS_EPOCHS=2 \
  bash scripts/run_cr_hkge_planCplus_holdout_study.sh crhkgeCplus_fullkg
```

### Output files
| File | Meaning |
|---|---|
| `planCplus_holdout_study/logs/planCplus_invariant_check.log` | invariants (a)+(c) result |
| `planCplus_holdout_study/logs/<cell>_seed<seed>_train.log` | training log (prints contrastive pair count + λ_c/margin) |
| `planCplus_holdout_study/logs/<cell>_seed<seed>_subset_eval.log` | overall/enriched/standard Top-K |
| `planCplus_holdout_study/logs/planCplus_holdout_summary_meanstd.{md,csv}` | aggregated 2×2 mean±std (CSV matches `holdout_summary_meanstd.csv`) |
| `planCplus_holdout_study/<cell>/seed_<seed>/weights/...` | per-cell, per-seed checkpoints |
