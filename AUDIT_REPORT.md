# CR-HKGE Static Code Audit

**Scope:** Static read-only audit of the CR-HKGE recommender (KGAT-based, 340-product
fragrance catalog, zero real user interactions, content-based surrogate supervision).
**No training was run and nothing was modified.** All findings are traced to source.

**Files traced:**
`Model/CRHKGE.py`, `Model/KGAT.py`, `Model/Main.py`, `Model/evaluate_item_subsets.py`,
`Model/utility/loader_kgat.py`, `Model/utility/load_data.py`, `Model/utility/parser.py`,
`Model/CFKG.py`, `Model/utility/loader_cfkg.py`,
`scripts/build_cr_hkge_ready_dataset.py`, `scripts/build_aromatique_dataset_variants.py`,
the run scripts (`scripts/run_cr_hkge_*.sh`, `scripts/run_corrected_kgat_baselines.sh`),
`IEEE-CRHKGE.md` (the paper brief), and the three dataset folders.

Severity legend: **CRITICAL** = undermines the central claim/validity; **MODERATE** =
materially weakens a claim or fairness; **MINOR** = correctness/clarity issue.

---

## PRIORITY 1 — Is the evaluation circular? (most important)

### 1.1 Signals that construct the positive-pair labels

The labels (`train.txt`/`test.txt` positive pairs) are built by
`score_pair()` in [build_cr_hkge_ready_dataset.py:191](scripts/build_cr_hkge_ready_dataset.py:191),
summing these weighted components ([build_cr_hkge_ready_dataset.py:41](scripts/build_cr_hkge_ready_dataset.py:41)):

| Signal (label term) | Weight | How it is computed | Underlying KG relation(s) |
|---|---:|---|---|
| `local_accord` | 2.00 | Jaccard of two products' local accords | `has_accord` |
| `visual_note` | 1.25 | Jaccard of visual notes | `has_visual_note` |
| `local_family` | 1.50 | shared local family (set intersection) | `belongs_to_family` |
| `global_accord` | 2.00 | Jaccard of global accords | `has_global_accord` (via `inspired_by`) |
| `global_family` | 1.25 | Jaccard of global families | `belongs_to_global_family` (via `inspired_by`) |
| `cross_ref_global_to_local_accord` | 1.75 | max Jaccard(global accords ↔ other's local accords) | `inspired_by` + `has_global_accord` + `has_accord` |
| `cross_ref_global_to_local_family` | 0.80 | same, families | `inspired_by` + global/local family |
| `same_global_reference` | **3.00** | both share a global reference node | `inspired_by` (same tail) |
| `sem_similar` | 0.35 | one is in the other's `sem_similar` set | `sem_similar` |
| `enriched_cross_ref_bonus` | 0.20 | bonus if both enriched and any global/cross term fired | `inspired_by` + globals |

Every label term is a deterministic function of the **exact same seven relations**
that live in `kg_final.txt`
([build_product_features](scripts/build_cr_hkge_ready_dataset.py:143) reads
`has_accord`, `has_visual_note`, `belongs_to_family`, `inspired_by`,
`has_global_accord`, `belongs_to_global_family`, `sem_similar`).

### 1.2 Does the model consume those same signals in its forward pass?

Yes — all of them, by design.

- **The graph the model propagates over is built directly from `kg_final.txt`.**
  `KGAT_loader._get_relational_adj_list()`
  ([loader_kgat.py:38](Model/utility/loader_kgat.py:38)) turns every triple into a
  (forward + inverse) adjacency block; these become the Laplacian `A_in` used in
  message passing. So `has_accord`, `has_global_accord`, `belongs_to_family`,
  `belongs_to_global_family`, `sem_similar`, and `inspired_by` are **propagation
  paths** in the model. Two products sharing an accord are 2 hops apart
  (product → accord → product) and get pulled together in embedding space — which is
  exactly the `local_accord`/`global_accord` label terms (weights 2.0).

- **`inspired_by` / same-global-reference is the explicit novelty path.**
  `_build_cr_hkge_data()` ([loader_kgat.py:286-322](Model/utility/loader_kgat.py:286))
  builds `product_global_mat` from `inspired_by` edges, and
  `_create_cross_reference_context()`
  ([CRHKGE.py:398-428](Model/CRHKGE.py:398)) propagates the global-reference node's
  attributes back into the product. Two products that share a global reference both
  receive context from that shared node, pulling their embeddings together. This is
  the **highest-weighted label term** (`same_global_reference` = 3.00) and the second
  group (`cross_ref_global_to_local_*` = 1.75/0.80, `global_accord` = 2.00).

- **Relation-type attention (`λ_r`) up-weights exactly the label-defining relations.**
  The fragrance prior ([CRHKGE.py:170-179](Model/CRHKGE.py:170)) initializes
  `sem_similar`=3.0, `has_accord`=2.0, `inspired_by`=1.8, `belongs_to_family`=1.8,
  i.e. it boosts the same relations that carry the label weight.

So **global-accord, global-family, same-global-reference, and cross-reference signals
appear in BOTH the label construction AND the model's forward pass.** This is not an
incidental overlap; the dataset builder's own docstring says the split is built "to be
aligned with the three CR-HKGE novelties"
([build_cr_hkge_ready_dataset.py:3-9](scripts/build_cr_hkge_ready_dataset.py:3)).

### 1.3 Train/test leakage

Three distinct leakage channels:

1. **Train and test positives are siblings from one ranked list.** For each profile,
   candidates are scored, sorted, and the top `train_per_profile + test_per_profile`
   (8 + 4 = 12) are taken; `split_ranked_candidates()`
   ([build_cr_hkge_ready_dataset.py:238-272](scripts/build_cr_hkge_ready_dataset.py:238))
   then **interleaves** them into train/test by a fixed stride. Test positives are
   therefore the *same kind of top-ranked content neighbor* as the training positives,
   drawn from the identical formula. A model fit to rank the 8 train neighbors high
   will rank the 4 test neighbors high automatically.

2. **The label-generating information is fully visible at training time.**
   `kg_final.txt` is loaded in full and is used for both message passing and the TransR
   KGE objective; **no test-relevant edge is held out or masked.** Everything needed to
   reconstruct a test label is an input the model trains on.

3. **The labels are literally the formula's nearest neighbors.** Because positives are
   the top-k of a fixed deterministic content-similarity function over the KG, the
   ceiling of the task is "reproduce that function." Any non-learned content kNN over
   the same features would score near the top. Models are competing on *fidelity to a
   fixed formula*, not on predicting held-out preference.

### 1.4 Verdict (Priority 1)

**The evaluation is circular. — CRITICAL.**
The positive-pair labels are a deterministic function of the same seven KG relations
that the model ingests as propagation paths, and the model's headline novelties
(cross-reference via `inspired_by`, relation-type priors) target precisely the
highest-weighted label terms (`same_global_reference`=3.0, `cross_ref_*`=1.75,
`global_accord`=2.0). Train and test positives come from the same ranked neighbor list
of that formula, with the full KG visible during training. The reported "improvement"
of CR-HKGE over KGAT is best interpreted as **CR-HKGE approximating the label-generating
content formula more faithfully than plain KGAT**, not as evidence of better
recommendation. In particular, the enriched-product gain (Section 9.3 of the brief)
is an expected artifact: the cross-reference signal is injected into the model *and* is
the dominant label signal for enriched products.

> Mitigations that would break the circularity (none currently present): hold out a
> *relation type* from the KG that also defines the label (e.g., train without
> `inspired_by`/global edges, test on same-global-reference labels); or obtain labels
> from a source independent of `kg_final.txt` (real interactions, human relevance
> judgments). The "attribute-only" KGAT variants in
> [build_aromatique_dataset_variants.py](scripts/build_aromatique_dataset_variants.py)
> reduce the *baseline's* access to novelty relations but do **not** remove the
> circularity, because the labels are still built from those relations.

---

## PRIORITY 2 — Is it truly zero-interaction?

### 2.1 No real user-item history is used — but the architecture still has a "user" table

- The dataset has no purchase/click/rating logs. In the CR-HKGE-ready set, each line of
  `train.txt`/`test.txt` is a **content profile = one source product**
  (`profile2product.tsv` shows `profile_id == source_product_id`, 0..339), and the
  "positive items" are content neighbors. Confirmed: brief says use the terms
  "content-based positive pairs / surrogate supervision / zero historical interaction"
  ([IEEE-CRHKGE.md:707-708](IEEE-CRHKGE.md:707)).

- **However, the model is not user-free.** `load_data._statistic_ratings()`
  ([load_data.py:61-63](Model/utility/load_data.py:61)) sets `n_users = max(profile_id)+1
  = 340`, and `KGAT._build_weights()` allocates a learned
  `user_embed` of shape `[340, 64]` ([KGAT.py:121](Model/KGAT.py:121)). Prediction is
  `u_e · p_e` ([KGAT.py:177](Model/KGAT.py:177)). So the model trains **one profile
  embedding per source product** and recommends via profile→product inner product. It is
  "interaction-free," but it is *not* true to say it "uses no user embeddings": it
  repurposes KGAT's user slot as a per-product profile slot. The brief's framing is
  defensible (these are not historical users), but a reviewer may object that the design
  is effectively item→item similarity learned through an extra embedding table.
  **Severity: MINOR** (framing/disclosure, not a correctness bug).

### 2.2 The legacy `dataset-aromatique-kgat-ready` set DOES contain 68 "users"

`dataset-aromatique-kgat-ready/summary.json` reports `n_train_users=68`,
`n_train_interactions=1490`, `n_test_interactions=406`, and the raw
`dataset-aromatique/train.txt` likewise has 68 user rows. These are **not** the
content-profile construction; they are an earlier interaction-style split.
Crucially, the **run scripts default to this 68-user dataset**, not the
zero-interaction one:

- `run_cr_hkge_final_study.sh`: `DATASET="${DATASET:-dataset-aromatique-kgat-ready}"`
  ([run_cr_hkge_final_study.sh:7](scripts/run_cr_hkge_final_study.sh:7))
- `run_cr_hkge_ablation.sh`: `--dataset dataset-aromatique-kgat-ready` (hardcoded)
  ([run_cr_hkge_ablation.sh:13](scripts/run_cr_hkge_ablation.sh:13))

…whereas the paper brief states the final model uses
`dataset-aromatique-crhkge-ready` ([IEEE-CRHKGE.md:25](IEEE-CRHKGE.md:25)). To reproduce
the paper's "zero-interaction" numbers you must override `DATASET=` — nothing in the
committed scripts does so, and there are no committed result logs (see Priority 6.3).
**Severity: MODERATE** — the default experiment path does not match the paper's stated
zero-interaction dataset, and the provenance of the 68 "users" is undocumented.

### 2.3 Modules that assume users/sequences

No module assumes interaction sequences or temporal order. `user_embed` is the only
user-shaped structure and it is a static embedding table. CFKG additionally treats the
profile→item link as relation 0 in a TransE graph ([loader_cfkg.py:51-53, 277](Model/utility/loader_cfkg.py:51)).
No leakage from sequence modeling. **OK.**

---

## PRIORITY 3 — Formula correctness vs KGAT

### 3.1 Prediction: inner product, not cosine

Training and offline evaluation use a plain **inner product**:

```python
# KGAT.py:177
self.batch_predictions = tf.matmul(self.u_e, self.pos_i_e, transpose_a=False, transpose_b=True)
# KGAT.py:213-214 (BPR)
pos_scores = tf.reduce_sum(tf.multiply(self.u_e, self.pos_i_e), axis=1)
neg_scores = tf.reduce_sum(tf.multiply(self.u_e, self.neg_i_e), axis=1)
```

`evaluate_item_subsets.py` scores via `model.eval()` → `batch_predictions`
([evaluate_item_subsets.py:221](Model/evaluate_item_subsets.py:221)), i.e. inner
product. Note each layer's embedding is L2-normalized then **concatenated** across
layers ([KGAT.py:292-296](Model/KGAT.py:292), [CRHKGE.py:355-358](Model/CRHKGE.py:355)),
so the final vector is a concatenation of unit-norm sub-vectors — neither raw dot nor
true cosine.

**Mismatch:** the exported serving config declares `"score_function": "cosine"`
([CRHKGE.py:687](Model/CRHKGE.py:687)). So the deployed retrieval ranks by cosine while
the model was trained/evaluated by inner product → **train/serve skew**.
- (a) Code: offline = inner product; export config = cosine.
- (b) Claim match: the paper's offline metrics are inner-product based (consistent
  internally); the serving system would behave differently from what was evaluated.
- (c) **Severity: MODERATE** for the conversational-serving claim; does not invalidate
  the offline tables.

### 3.2 Product update: full Bi-Interaction (⊙ present), matches KGAT

`CRHKGE._create_bi_interaction_embed()` keeps both KGAT branches:

```python
# CRHKGE.py:343-352
add_embeddings = ego_embeddings + sum_side_embeddings
sum_embeddings = leaky_relu(add_embeddings @ W_gc + b_gc)          # GCN/additive branch
bi_embeddings  = tf.multiply(ego_embeddings, bi_side_embeddings)    # <-- element-wise ⊙
bi_embeddings  = leaky_relu(bi_embeddings @ W_bi + b_bi)
ego_embeddings = bi_embeddings + sum_embeddings
```

This is the genuine Bi-Interaction aggregator of KGAT (KDD'19), identical in structure
to the base ([KGAT.py:274-287](Model/KGAT.py:274)). The element-wise term is present.
**Matches the claim. Severity: OK.** (Requires `--alg_type bi`; the run scripts set
`--alg_type bi`. With the parser default `ngcf`, `_build_model_phase_I` would raise
`NotImplementedError` — see [KGAT.py:160-171](Model/KGAT.py:160) — so the scripts must,
and do, pass `bi`.)

### 3.3 Relation-type attention λ_r: two softmaxes, multiplicative composition

- λ_r is a learnable logit vector, softmaxed:
  ```python
  # CRHKGE.py:95-105
  all_weights['cr_relation_type_logits'] = tf.Variable(...)        # learnable λ_r
  self.cr_relation_type_probs = tf.nn.softmax(cr_relation_type_logits)   # SOFTMAX #1
  self.cr_relation_type_multipliers = self.cr_relation_type_probs * n_relation_types   # type_count scaling → mean 1
  ```
- It multiplies the TransR attention score **before** KGAT's neighbor softmax:
  ```python
  # CRHKGE.py:297-299
  kg_score = tf.reduce_sum(tf.multiply(t_e, tf.tanh(h_e + r_e)), 1)
  relation_multiplier = self._relation_multiplier_for_r(r)
  return kg_score * relation_multiplier
  ```
  and the neighbor softmax is KGAT's `tf.sparse.softmax`
  ([KGAT.py:382-385](Model/KGAT.py:382)), applied during `update_attentive_A`.
  So there **are two softmax layers**, composed as
  `neighbor_softmax_j( λ_{r(ij)} · π(h_i, r_ij, t_j) )`.

Two correctness notes:
- π = `t_e·tanh(h_e+r_e)` can be **negative**; multiplying a negative score by a larger
  λ makes it *more negative*, so "priority" is not a clean monotone reweighting of
  attention. The `type_count` scaling keeps the average multiplier at 1 so uniform λ
  reproduces KGAT (good design intent, [CRHKGE.py:101-105](Model/CRHKGE.py:101)).
  **Severity: MINOR** (interpretability of λ_r).
- **Gradient path:** the λ_r → attention → `A_in` route goes through
  `update_attentive_A()`, which runs a separate `sess.run` and rebuilds `A_in` as a
  SciPy matrix ([KGAT.py:438-468](Model/KGAT.py:438)) — **non-differentiable**. So λ_r
  does *not* receive gradient through its headline "reweight KGAT attention" channel; it
  learns only through the differentiable paths: relation-aware messages
  ([CRHKGE.py:393-394](Model/CRHKGE.py:393)) and the cross-reference `inspired_multiplier`
  / global-attribute attention ([CRHKGE.py:412-424](Model/CRHKGE.py:412)). The attention
  effect of λ_r is "open-loop" (affects forward, not its own gradient). **Severity:
  MODERATE** — the mechanism is active but optimized via a different channel than the
  paper's narrative implies.

### 3.4 TransR energy + KG loss + BPR loss

All match KGAT:
- KGE plausibility energy `‖W_r e_h + e_r − W_r e_t‖²` with BPR/softplus
  ([KGAT.py:233-241](Model/KGAT.py:233)); per-relation transform `trans_W`
  ([KGAT.py:137](Model/KGAT.py:137)) applied in `_get_kg_inference`.
- Attention score = `(W_r e_t)·tanh(W_r e_h + e_r)` ([KGAT.py:409](Model/KGAT.py:409)).
- CF BPR loss = softplus on inner-product score difference ([KGAT.py:220](Model/KGAT.py:220)).

**Matches KGAT. Severity: OK.** CR-HKGE's only change is the λ_r multiplier on the
attention score and the additive cross-reference context.

---

## PRIORITY 4 — Do the claimed mechanisms actually work?

### 4.1 Cross-reference: `c_ref` from global node via `inspired_by`, zeroed for standard products

Confirmed working as described:

```python
# CRHKGE.py:415-428
global_reference_context = ego_embeddings + attr_context           # global ref node + its attributes
product_context = product_global_tensor @ global_reference_context  # flow global→product along inspired_by
transformed_context = leaky_relu(product_context @ W_cr + b_cr)
return (alpha * gate * inspired_multiplier * transformed_context * product_mask)
```

- `product_global_tensor` is built **only** from `inspired_by` edges
  ([loader_kgat.py:298-322](Model/utility/loader_kgat.py:298)), so standard products have
  zero rows there.
- `product_mask` is 1 **only** for enriched products
  ([loader_kgat.py:332-334](Model/utility/loader_kgat.py:332)); the final multiply by
  `self.cr_product_mask_tensor` forces `c_ref = 0` for standard products
  ([CRHKGE.py:428](Model/CRHKGE.py:428)). **Double-guaranteed zero for standard
  products. Matches claim. Severity: OK.**

### 4.2 Relation prior / λ_r: learnable, active, and exported

- Learnable: `cr_relation_type_logits` is a `tf.Variable`
  ([CRHKGE.py:95](Model/CRHKGE.py:95)).
- Active in loss: yes via the differentiable paths in §3.3 (when cross-reference and/or
  relation-aware message are on — both on in the final model). **Caveat from §3.3:** the
  attention-reweighting channel itself is non-differentiable.
- Exported after training: yes — `export_artifacts()` →
  `_relation_weight_rows()` writes per-relation `probability`, `multiplier`,
  `message_multiplier` to `relation_weights.tsv`
  ([CRHKGE.py:528-549, 615-625](Model/CRHKGE.py:528)). **Matches claim. Severity: OK.**

### 4.3 Is `alpha = 0.1` hardcoded in the right place?

- Applied in the right place: `alpha` scales the cross-reference context before it is
  added to the additive branch ([CRHKGE.py:339, 427](CRHKGE.py:427)). Correct location.
- **Not hardcoded to 0.1.** It is a CLI argument `--cr_cross_ref_alpha` whose **parser
  default is 1.0** ([parser.py:97](Model/utility/parser.py:97)); the separate "pure"
  trainer defaults it to 0.5 ([train_cr_hkge_pure.py:434](scripts/train_cr_hkge_pure.py:434)).
  The value 0.1 only appears for the `cr_hkge_final_alpha_0_1` target in
  `run_cr_hkge_final_study.sh` ([line 63](scripts/run_cr_hkge_final_study.sh:63)).
  This is fine, but note the paper's "final" α=0.1 is **not** the default any script
  runs unless that specific target is selected. **Severity: MINOR** (reproducibility
  clarity).

> Note: there are **two CR-HKGE implementations** — `Model/CRHKGE.py` (used by
> `Main.py`, the documented final path) and a separate NumPy-style trainer
> `scripts/train_cr_hkge_pure.py`. They must be kept in sync; the paper's final variant
> name maps to the `Main.py`/`CRHKGE.py` path.

---

## PRIORITY 5 — Ablation validity

The ablation flags genuinely disable the named components (not renames):

| Variant | Flag effect (verified) | Disables claimed component? |
|---|---|---|
| `no_cross_reference` / `A1` | `cr_use_cross_ref 0` → `_create_cross_reference_context` not called; `cr_cross_ref_gate=0` ([CRHKGE.py:152-153, 334](Model/CRHKGE.py:152)) | ✅ cross-reference off |
| `no_fragrance_prior` | `cr_relation_prior_mode none` → logits init to **zeros** ([CRHKGE.py:158-159](Model/CRHKGE.py:158)); λ_r still learnable | ✅ prior off (weights still trainable) |
| `no_relation_attention` / `A_no_novelty` | `cr_use_relation_weight 0` → all multipliers = 1 ([CRHKGE.py:120-123, 256-257](Model/CRHKGE.py:120)) | ✅ relation attention off |
| `no_relation_message` / `A2` | `cr_relation_aware_message 0` → falls back to KGAT `A_in` propagation ([CRHKGE.py:367-379](Model/CRHKGE.py:367)) | ✅ message reweight off |
| `no_novelty_modules` / `A4` | both flags 0 → `_create_bi_interaction_embed` returns `super()` = vanilla KGAT ([CRHKGE.py:302-303](Model/CRHKGE.py:302)) | ✅ reduces to KGAT |

So the disabling is real. **But two validity concerns:**

1. **Inconsistent α between the ablation variants and the reported "full" model — MODERATE.**
   In `run_cr_hkge_final_study.sh`, the headline full model `cr_hkge_final` uses
   `--cr_cross_ref_alpha 0.5` ([line 51](scripts/run_cr_hkge_final_study.sh:51)), the
   paper's reported full model uses **α=0.1** ([IEEE-CRHKGE.md:888](IEEE-CRHKGE.md:888)),
   yet `A_no_fragrance_prior` is run with **α=0.5**
   ([line 122](scripts/run_cr_hkge_final_study.sh:122)). Comparing an α=0.5 ablation
   against an α=0.1 "full" changes **two** variables at once, so the attributed effect of
   "fragrance prior" is confounded with the α change. (For `no_cross_reference`, α is moot
   since cross-ref is off.)

2. **`relation_aware_message` is not held constant across ablations — MODERATE.**
   `A_no_cross_reference` keeps `--cr_relation_aware_message 1`
   ([line 107](scripts/run_cr_hkge_final_study.sh:107)) while `A_no_relation_attention`
   and `A_no_novelty_modules` set it to 0. The "remove cross-reference" ablation therefore
   differs from "remove all novelty" in *two* components, not one.

3. **Layer depth is held constant — OK.** All variants (and baselines) use
   `--layer_size [64,32,16]` (3 layers) in `COMMON_ARGS`
   ([run_cr_hkge_final_study.sh:19](scripts/run_cr_hkge_final_study.sh:19),
   [run_cr_hkge_ablation.sh:18](scripts/run_cr_hkge_ablation.sh:18)). Seeds, lr, epochs,
   dropout are also shared via `COMMON_ARGS`. So multi-layer effects do **not**
   differentially contaminate attribution across variants — good.

---

## PRIORITY 6 — Baselines & reproducibility

### 6.1 CFKG is not configured comparably and shows non-convergence — MODERATE/CRITICAL for that baseline

- CFKG optimizes with **plain SGD** (`GradientDescentOptimizer`)
  ([CFKG.py:142](Model/CFKG.py:142)) at `lr=0.0001`, while KGAT/CR-HKGE/BPRMF/CKE use
  **Adam** at the same `lr` ([KGAT.py:230](Model/KGAT.py:230)). A learning rate tuned for
  Adam (1e-4) is far too small for vanilla SGD with a margin-ranking loss
  ([CFKG.py:129](Model/CFKG.py:129)); on this tiny catalog and ~24-43 batches/epoch ×100
  epochs the model barely moves from Xavier init.
- The brief's own table reports CFKG at **Recall@3 = 0.00735, NDCG@3 = 0.02027**
  ([IEEE-CRHKGE.md:797](IEEE-CRHKGE.md:797)) — essentially random — and the brief then
  uses this to argue "simple collaborative KG modeling is insufficient"
  ([IEEE-CRHKGE.md:808](IEEE-CRHKGE.md:808)).
- (a) Code: CFKG uses SGD + tiny lr → undertrained. (b) Claim match: the near-zero
  result is most plausibly an **optimization/convergence artifact**, not evidence about
  KG-collaborative modeling. (c) **Severity: MODERATE** (a published near-zero baseline
  attributed to the method rather than to its mis-tuned optimizer is a real review risk).

### 6.2 Seeds fixed, but single run only — MODERATE

- Seeds are fixed: `tf.set_random_seed(2019); np.random.seed(2019); rd.seed(2019)`
  ([Main.py:67-69](Model/Main.py:67)); evaluator sets tf+np seeds
  ([evaluate_item_subsets.py:249-250](Model/evaluate_item_subsets.py:249)).
- Each target is trained **once** (the run scripts loop targets, not repeats), and the
  brief reports point estimates with **no standard deviation / no multi-seed runs**. The
  KGAT→CR-HKGE gains are small (e.g., NDCG@3 +0.038, Recall@10 +0.023;
  [IEEE-CRHKGE.md:814-821](IEEE-CRHKGE.md:814)) and unaccompanied by variance, so
  significance cannot be assessed. **Severity: MODERATE.**

### 6.3 Hyperparameters match the brief; no committed result logs — MINOR/MODERATE

- Code vs brief §8.3: embed 64, kge 64, layers [64,32,16], lr 1e-4, batch 64,
  node_dropout [0.1], mess_dropout [0.1,0.1,0.1], epochs 100, Ks [3,5,10],
  best NDCG@3, prior strength 1.0 — **all match** `COMMON_ARGS`. ✅
- But **no experiment logs are committed** (only legacy
  `Log/training_log_amazon-book.log` and `last-fm` exist; no `final_study/`,
  `ablation/`, or `corrected_baselines/` outputs). The exact numbers in the brief's
  tables cannot be traced to a committed run, and (see §2.2) the scripts default to the
  68-user dataset, not `crhkge-ready`. **Severity: MODERATE** for reproducibility.

### 6.4 Paper-comparison script compares across different datasets — MODERATE

`run_cr_hkge_paper_comparison.sh` trains the KGAT baseline on
`dataset-aromatique-attribute-kgat-ready` (local attributes only, no cross-ref/sem_similar)
and CR-HKGE on `dataset-aromatique-kgat-ready` (full KG)
([run_cr_hkge_paper_comparison.sh:6-17](scripts/run_cr_hkge_paper_comparison.sh:6)). The
intent (documented in [build_aromatique_dataset_variants.py:1-19](scripts/build_aromatique_dataset_variants.py:1))
is to stop a vanilla KGAT baseline from "consuming the novelty." That is a defensible
*diagnostic*, but **as a headline comparison it is not apples-to-apples**: the two models
see different graphs and (unless `--cr_subset_dataset` aligns them) potentially different
test positives. The subset evaluation does re-derive enriched/standard from a common
`dataset-aromatique-kgat-ready` ([line 53-54](scripts/run_corrected_kgat_baselines.sh:53)),
but the *training graph* differs. **Severity: MODERATE** — if any paper number uses this
script, the baseline was deliberately handicapped.

---

## PRIORITY 7 — Dataset construction

Verified counts against `dataset-aromatique-crhkge-ready` (the paper's stated dataset):

| Item | Claimed | Actual (verified) | Match |
|---|---|---|---|
| Products | 340 | 340 (`product2id.tsv`, profiles 0..339) | ✅ |
| Entities | 998 | 998 (`entity2id_typed.tsv`, ids 0..997) | ✅ |
| Relations | 7 | 7 (`relation2id.txt`) | ✅ |
| KG triples | 9250 **or** 9255? | **9250** (`wc -l kg_final.txt` = 9250; per-relation 243+1629+680+340+4110+2017+231 = 9250) | 9250 ✅ |
| Enriched | 243 | 243 (= `inspired_by` count; one edge per enriched product) | ✅ |
| Standard | 97 | 97 (= 340 − 243) | ✅ |

- **The "9255" figure is incorrect; the actual triple count is 9250.** Both the file
  line count and the relation-wise sum agree on 9250 (also matches both `summary.json`
  files). If the paper cites 9255, that is a typo. **Severity: MINOR.**
- **Enriched/standard split is derived correctly from `inspired_by` presence.**
  `ProductFeature.is_enriched = bool(self.global_refs)` where `global_refs` is populated
  only on the `inspired_by` branch
  ([build_cr_hkge_ready_dataset.py:76-77, 181-184](scripts/build_cr_hkge_ready_dataset.py:76));
  the evaluator independently re-derives the same definition from `kg_final.txt`
  ([evaluate_item_subsets.py:57-80](Model/evaluate_item_subsets.py:57)). Consistent. ✅
- **Data leakage between train/test positives:** as detailed in §1.3, train and test
  positives for a profile are the same top-12 content neighbors split by stride — they
  are not independent. Within a single profile a product cannot be both train and test
  positive (the 12 are partitioned), but across the dataset the train/test sets are
  generated by the identical formula over the identical, fully-visible KG. **This is the
  core leakage and is CRITICAL** (counted under Priority 1).
- Minor robustness note: the builder requires ≥12 scoring candidates per profile or it
  raises ([build_cr_hkge_ready_dataset.py:245-248](scripts/build_cr_hkge_ready_dataset.py:245));
  with `min_score=0.01` and a 340-product catalog this holds, but it means every product
  is forced to have 4 "test positives" even when genuine similarity is weak (mean score
  3.93, min 1.66 per `summary.json`). **Severity: MINOR.**

---

## Summary Table

| # | Finding | Matches claim? | Severity |
|---|---|---|---|
| 1 | **Labels and model inputs are the same 7 KG relations; novelties target the top label weights; train/test positives are siblings of one formula-ranked list with full KG visible** | No — evaluation is self-fulfilling | **CRITICAL** |
| 2 | CFKG uses SGD@1e-4 while others use Adam → near-zero result is an optimizer/convergence artifact, presented as a method conclusion | No | **MODERATE** |
| 3 | Run scripts default to the 68-user `kgat-ready` set; paper claims `crhkge-ready`; no committed logs to confirm which produced the numbers | Unclear / undocumented | **MODERATE** |
| 4 | Ablation confounds: `no_fragrance_prior` run at α=0.5 vs full α=0.1; `relation_aware_message` not held constant across ablations | Partially | **MODERATE** |
| 5 | λ_r's headline "reweight KGAT attention" channel is non-differentiable (`update_attentive_A` rebuilds `A_in` in NumPy); λ_r trains only via message/cross-ref paths | Partially | **MODERATE** |
| 6 | Paper-comparison script trains baseline and CR-HKGE on different graphs | Diagnostic only, not fair headline | **MODERATE** |
| 7 | Single run, fixed seed, no variance reported; gains are small (NDCG@3 +0.038) | N/A | **MODERATE** |
| 8 | Offline ranks by inner product; export/serving config says cosine (train/serve skew) | No | **MODERATE** |
| 9 | Model keeps a 340-row `user_embed` (profile-per-product); "no user embeddings" framing is loose | Mostly | **MINOR** |
| 10 | α is a CLI arg (default 1.0/0.5), not hardcoded 0.1; correct *location* though | Mostly | **MINOR** |
| 11 | λ_r multiplies a possibly-negative attention score → "priority" not strictly monotone | Mostly | **MINOR** |
| 12 | Triple count is 9250, not 9255 | Typo if 9255 cited | **MINOR** |
| 13 | Bi-Interaction (⊙) correctly implemented; TransR/KGE/BPR match KGAT; cross-ref correctly zeroed for standard products; enriched/standard split correct; λ_r exported | Yes | **OK** |

---

## Overall Verdict

**The implementation is, at the code level, a faithful and competent extension of KGAT**
— the Bi-Interaction aggregator, TransR KGE, BPR objective, cross-reference propagation,
and the enriched/standard machinery all do what the brief says, and the ablation flags
genuinely toggle the components they name.

**However, the central scientific claim — that CR-HKGE delivers better recommendation —
is not supported by the evaluation as constructed, because the evaluation is circular
(Finding 1, CRITICAL).** The positive-pair labels are a deterministic function of the
exact KG relations the model propagates over, the model's signature novelties
(cross-reference via `inspired_by`, relation priors) target the highest-weighted label
terms, and train/test positives are drawn from one formula-ranked neighbor list with the
full label-generating KG visible during training. Under these conditions, "beating KGAT"
means "approximating the fixed content-similarity formula slightly better," not
"recommending better." The enriched-product gain in particular is an expected
consequence of feeding the dominant enriched-product label signal into the model.

Secondary issues compound the risk for a paper: a mis-tuned CFKG baseline whose
near-zero score is over-interpreted (Finding 2), confounded ablations (Finding 4), a
non-differentiable λ_r-attention channel that complicates the relation-attention
narrative (Finding 5), cross-dataset baseline comparison (Finding 6), single-run
point estimates with small gains and no variance (Finding 7), default scripts pointing
at a different dataset than the paper claims with no committed logs (Findings 3), and an
inner-product/cosine train-serve skew (Finding 8).

**Recommendation:** Do not present the current Top-K numbers as evidence of
recommendation quality without (a) breaking the label/model relation overlap — e.g.,
hold out `inspired_by`/global edges from the training graph and test on
same-global-reference labels, or obtain labels independent of `kg_final.txt`; (b)
re-tuning CFKG (Adam or a larger SGD lr) before drawing conclusions about it; (c)
aligning α and `relation_aware_message` across ablation variants; (d) reporting
multi-seed mean±std; and (e) confirming and committing the exact run logs on
`dataset-aromatique-crhkge-ready`. Reframing the contribution honestly as
*content-aware retrieval / offline surrogate ranking* (which the brief's own wording
gestures toward) rather than *interaction recommendation* would also align the claims
with what the code actually measures.

*Static audit only — no training was performed and no files other than this report were
created or modified.*
