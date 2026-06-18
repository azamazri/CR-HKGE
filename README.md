# CR-HKGE

Clean research repository for CR-HKGE: Cross-Reference Heterogeneous Knowledge Graph Embedding for fragrance recommendation.

This repository is a focused extraction from the larger thesis workspace. It keeps the CR-HKGE model, Aromatique datasets, reproducibility scripts, paper assets, and core research notes. It intentionally excludes the external application code, benchmark datasets, runtime logs, generated weights, and serving integration files.

## What Is Included

- `Model/`
  - `CRHKGE.py`: proposed CR-HKGE model.
  - `KGAT.py`: KGAT architectural baseline.
  - `BPRMF.py`, `CKE.py`, `CFKG.py`, `NFM.py`: local baseline implementations inherited from the KGAT codebase.
  - `evaluate_item_subsets.py`: overall, enriched, and standard subset evaluation.
  - `utility/`: loaders, parser, metrics, TensorFlow compatibility helpers.
- `dataset-aromatique-crhkge-ready/`
  - Final CR-HKGE-ready dataset with content-based positive pairs.
- `dataset-aromatique-kgat-ready/`
  - Intermediate KGAT-compatible Aromatique dataset used as a source for CR-HKGE-ready construction.
- `dataset-aromatique/`
  - Original local Aromatique KG files.
- `scripts/`
  - Dataset construction, training, ablation, summarization, retrieval, and audit scripts.
- `paper_assets/`
  - Figure-generation assets for the IEEE paper.
- `reference/`
  - Core research blueprint, validation notes, implementation guide, progress notes, and paper draft.
- `IEEE-CRHKGE.md`
  - Detailed IEEE paper writing brief.
- `AUDIT_REPORT.md`
  - Static code audit and validity notes.
- `CR-HKGE_IEEE_Conference_Paper.docx` and `.pdf`
  - Current paper draft outputs.

## What Is Excluded

- `_external/`: application repository and Supabase integration.
- `Data/`: original KGAT benchmark datasets such as Amazon-Book, Yelp, and Last-FM.
- `Log/`, `weights/`, `artifacts/`, `final_study/`, runtime checkpoints, and generated experiment outputs.
- Non-CR-HKGE reference files such as XAI prompt experiments and store-availability pilot notes.

## Research Scope

CR-HKGE adapts KGAT to a fragrance recommendation setting where the Aromatique catalog has no historical purchase, rating, or click interactions. The training split uses content-based positive pairs as surrogate supervision.

The three CR-HKGE novelty components are:

1. Fragrance-specific heterogeneous KG construction.
2. Cross-reference semantic propagation via `inspired_by`.
3. Relation-type specific attention.

For the IEEE 5-page paper, the intended comparison is:

- BPRMF
- CKE
- CFKG
- KGAT
- CR-HKGE

NFM code is retained because it is part of the inherited local training code, but it is not part of the current IEEE paper discussion.

## Dataset Summary

Final dataset:

```text
dataset-aromatique-crhkge-ready
products: 340
enriched products: 243
standard products: 97
entities: 998
relation types: 7
KG triples: 9,250
train positive pairs: 2,720
test positive pairs: 1,360
```

Important wording for papers and reports:

```text
The dataset does not contain historical user interaction.
train.txt and test.txt contain content-based positive pairs for surrogate supervision.
```

## Colab Setup

From repository root in Colab:

```bash
bash scripts/setup_colab_env.sh
```

If TensorFlow import fails because of a JAX / `ml_dtypes` conflict, rerun the setup script and restart the runtime.

## Build CR-HKGE-Ready Dataset

```bash
python scripts/build_cr_hkge_ready_dataset.py
```

This regenerates `dataset-aromatique-crhkge-ready` from `dataset-aromatique-kgat-ready`.

## Run Final Study

Example final comparison:

```bash
DATASET=dataset-aromatique-crhkge-ready \
LOG_DIR=/content/kgat/final_study/logs_crhkge_ready \
bash scripts/run_cr_hkge_final_study.sh \
  kgat \
  bprmf \
  cke \
  cfkg \
  cr_hkge_final_alpha_0_1
```

Summarize logs:

```bash
python scripts/summarize_final_study.py /content/kgat/final_study/logs_crhkge_ready/*_subset_eval.log
```

## Run CR-HKGE Ablation

```bash
DATASET=dataset-aromatique-crhkge-ready \
LOG_DIR=/content/kgat/final_study/logs_crhkge_ready \
bash scripts/run_cr_hkge_final_study.sh \
  cr_hkge_final_alpha_0_1 \
  A_no_cross_reference \
  A_no_relation_attention \
  A_no_fragrance_prior \
  A_no_novelty_modules
```

## Paper Files

- Main brief: `IEEE-CRHKGE.md`
- Current draft: `CR-HKGE_IEEE_Conference_Paper.docx`
- Current PDF: `CR-HKGE_IEEE_Conference_Paper.pdf`
- Figures: `paper_assets/`

Use `AUDIT_REPORT.md` before making strong claims. It documents validity risks around content-based surrogate labels and KG-feature overlap.

## Notes

This repository is for research reproducibility and paper preparation. The external Aromatique application and deployment assets are intentionally not included.
