# dataset-aromatique-crhkge-holdout

Dataset ini dibuat oleh `scripts/build_cr_hkge_ready_dataset.py`.

## HOLD-OUT MODE (circularity-breaking)

`kg_final.txt` di sini adalah TRAINING GRAPH yang sudah dipangkas: edge
cross-reference langsung (inspired_by, has_global_accord, belongs_to_global_family) DIHAPUS, sedangkan relasi konten tidak
langsung (has_accord, has_visual_note, belongs_to_family, sem_similar) tetap dipertahankan.

- Edge dihapus total: 2491 (dari 9250 edge full KG).
- Edge tersisa untuk training: 6759.
- `train.txt`/`test.txt` IDENTIK dengan dataset full-KG (label tidak diubah).
- Saat evaluasi subset, gunakan `--cr_subset_dataset dataset-aromatique-crhkge-ready`
  agar definisi enriched/standard tetap berasal dari KG penuh.

Tujuan utamanya adalah menyelaraskan `train.txt` dan `test.txt` dengan tiga novelty CR-HKGE:

1. Fragrance-specific heterogeneous KG construction.
2. Cross-reference via `inspired_by`.
3. Relation-type priority/attention.

Format file tetap KGAT-compatible, sehingga KGAT, CR-HKGE, dan baseline lain dapat dibandingkan pada split yang sama.

Setiap profile merepresentasikan satu produk sumber/query content. Item positif adalah produk lain dengan skor relevance tertinggi berdasarkan kombinasi local fragrance attributes, global reference enrichment, cross-reference bridge, dan weak `sem_similar` support.

File penting:

- `train.txt`: positive pairs untuk BPR training.
- `test.txt`: held-out positive pairs untuk evaluasi Top-K.
- `profile2product.tsv`: mapping profile ke produk sumber.
- `positive_pair_scores.tsv`: audit skor setiap pasangan positif.
- `summary.json`: statistik dataset dan bobot scoring.
