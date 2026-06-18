# CR-HKGE Progress and Finalization Readiness

Tanggal konteks: 2026-06-03  
Workspace: `D:\Azam\MTI\TESIS\knowledge_graph_attention_network`  
Dokumen acuan wajib:

```text
03_Research_Azam_Blueprint.md
04_Joint_Research_Azam_Raissa.md
07_CR_HKGE_Implementation_Guide.md
```

Dokumen ini mencatat progres CR-HKGE yang sudah diimplementasikan, gap terhadap novelty yang tertulis di blueprint, dan instruksi teknis agar model benar-benar siap sebagai model utama paper conference. Dokumen ini tidak mengubah arah penelitian. Arah penelitian tetap mengikuti `03_Research_Azam_Blueprint.md` dan `04_Joint_Research_Azam_Raissa.md`.

---

## 1. Research Direction Lock

CR-HKGE tidak boleh digeser menjadi framing penelitian lain. Definisi yang dipakai tetap:

```text
Cross-Reference Semantic Enrichment on Heterogeneous Knowledge Graph Embedding
for Personalized Fragrance Recommendation in Conversational Systems
```

Kondisi masalah dari blueprint:

```text
1. Aromatique memiliki 340 produk aktif.
2. Aromatique tidak memiliki user interaction history.
3. Domain fragrance sangat terspesialisasi.
4. Produk lokal memiliki atribut revolutionize / inspired_by yang menghubungkan produk lokal ke parfum global.
5. Model KG recommender existing seperti KGAT membutuhkan user-item interaction matrix.
6. CR-HKGE dikembangkan untuk memanfaatkan heterogeneous fragrance KG dan cross-reference semantics.
```

Flow sistem yang harus dijaga:

```text
User input teks percakapan
  -> NLU Raissa
  -> structured query {accords, family, occasion}
  -> KG Storage Azam
  -> CR-HKGE Retrieval
  -> Top-3 products + KG paths
  -> XAI Raissa
  -> respons natural + rekomendasi + explanation
```

Jadi fokus finalisasi model adalah:

```text
CR-HKGE menghasilkan product embeddings dan KG-aware retrieval mechanism
yang dapat menerima structured query dari NLU dan mengeluarkan Top-3 + KG paths.
```

Bukan:

```text
Mengubah arah menjadi collaborative filtering user-history recommendation.
Mengubah arah menjadi penelitian murni query-profile tanpa KGAT basis.
Mengubah novelty setelah blueprint sudah dikunci.
```

---

## 2. Status Implementasi Saat Ini

### File Kode yang Sudah Ada

File baru:

```text
Model/CRHKGE.py
```

File yang sudah dimodifikasi:

```text
Model/Main.py
Model/utility/parser.py
Model/utility/batch_test.py
Model/utility/loader_kgat.py
```

Dokumentasi pendukung:

```text
07_CR_HKGE_Implementation_Guide.md
```

Commit GitHub penting:

```text
6b8e7e4 Add CR-HKGE model implementation
cc54bc8 Fix CR-HKGE relation adjacency slicing
d764086 Fix CR-HKGE artifact JSON export
```

### Fitur yang Sudah Berjalan

CR-HKGE saat ini sudah memiliki:

```text
1. model_type cr_hkge.
2. Loader metadata untuk CR-HKGE.
3. Deteksi produk enriched via relation inspired_by.
4. Mapping produk lokal -> global reference.
5. Mapping global reference -> global accord/global family.
6. Relation-type scalar weights.
7. Semantic relation tying untuk forward dan inverse relation.
8. Cross-reference context untuk produk dengan inspired_by.
9. Training CR-HKGE 100 epoch di Colab.
10. Export artifact product/entity embeddings, relation weights, kg_paths, model_config.
```

### Status Dataset yang Terbaca

Output loader CR-HKGE yang sudah benar:

```text
[n_users, n_items]=[68, 340]
[n_train, n_test]=[1490, 406]
[n_entities, n_relations, n_triples]=[998, 7, 9250]
CR-HKGE metadata: enriched_products=243, product_global_edges=243, global_attr_relations=2
```

Makna metadata:

```text
243 produk memiliki inspired_by / revolutionize.
243 product_global_edges berarti tiap enriched product punya global reference edge.
2 global_attr_relations berarti loader membaca has_global_accord dan belongs_to_global_family.
```

### Hasil Training Terakhir

Run CR-HKGE 100 epoch dengan:

```text
Ks = [3,5,10]
embed_size = 64
kge_size = 64
layer_size = [64,32,16]
alg_type = bi
adj_type = si
lr = 0.0001
```

Best result:

```text
Best Iter=[9]@[666.3]
recall    = [0.18104, 0.21739, 0.29326]
precision = [0.28431, 0.22941, 0.17059]
hit       = [0.52941, 0.52941, 0.57353]
ndcg      = [0.35522, 0.37067, 0.43844]
```

Karena `Ks=[3,5,10]`, maka:

```text
Recall@3    = 0.18104
Precision@3 = 0.28431
Hit@3       = 0.52941
NDCG@3      = 0.35522
```

Interpretasi sementara:

```text
Training sudah stabil.
Model sudah dapat mengeluarkan metrik Top-3.
Model belum boleh diklaim final sampai gap di bagian 4 diselesaikan.
```

---

## 3. Posisi Evaluasi 68 User dan 406 Test Interaction

Angka:

```text
n_users = 68
n_test = 406
```

harus dibaca sebagai konsekuensi format evaluasi KGAT-style yang sedang dipakai agar model dapat dilatih dan dievaluasi melalui `batch_test.py`.

Ini tidak mengubah fakta penelitian bahwa Aromatique tidak memiliki user interaction history. Dalam paper, jangan menulis seolah-olah ada 68 user historis nyata dari Aromatique.

Framing yang aman:

```text
The offline training and evaluation follow a KGAT-compatible interaction proxy generated from the Aromatique fragrance dataset, while the production retrieval stage uses CR-HKGE product embeddings with structured preference queries extracted from the conversational interface.
```

Atau dalam bahasa Indonesia:

```text
Evaluasi offline menggunakan format interaction proxy yang kompatibel dengan KGAT, sedangkan sistem akhir menggunakan structured preference query dari percakapan untuk melakukan retrieval terhadap product embeddings CR-HKGE.
```

Catatan penting:

```text
Jangan menggeser arah penelitian.
Jangan menyebut dataset memiliki real user history.
Jangan menghapus fakta bahwa problem utama adalah 0 interaction history.
Jelaskan bahwa adaptasi format KGAT dilakukan agar baseline KGAT/CR-HKGE dapat dieksekusi dan dievaluasi secara offline.
```

---

## 4. Gap Terhadap Blueprint CR-HKGE

Bagian ini adalah kritik teknis yang harus diselesaikan sebelum model dianggap final.

### Gap 1: Query Encoder Belum Final

Blueprint menyatakan:

```text
Encode query -> query vector
Cosine similarity dengan 340 product embeddings
Filter + ranking
Output Top-3 products + KG paths
```

Status sekarang:

```text
Training CR-HKGE sudah menghasilkan product/entity embeddings.
Namun query encoder dari structured query {accords, family, occasion} ke vector belum menjadi modul final.
```

Instruksi finalisasi:

```text
1. Ambil structured query dari NLU Raissa:
   {accords, family, occasion, intensity, gender/context jika ada}

2. Map setiap token preferensi ke entity KG:
   accords -> entity type accord/global_accord
   family -> entity type family/global_family
   notes -> entity type note jika tersedia

3. Ambil entity embeddings dari artifact CR-HKGE.

4. Bentuk query vector:
   q = weighted_mean(entity_embeddings)

5. Bobot query mengikuti relation-type weights:
   has_accord lebih dominan untuk accords
   belongs_to_family untuk family
   has_visual_note untuk visual note
   inspired_by hanya dipakai jika user menyebut reference/inspired perfume

6. Normalize q dan product embeddings.

7. Score:
   cosine(q, e_p*)

8. Ambil Top-3 product.
```

Output minimum query encoder:

```json
{
  "query_vector_dim": 176,
  "matched_entities": [
    {"text": "vanilla", "entity_id": 344, "entity_type": "accord", "relation": "has_accord"},
    {"text": "amber", "entity_id": 342, "entity_type": "family", "relation": "belongs_to_family"}
  ],
  "unmatched_terms": ["date night"]
}
```

### Gap 2: Embedding Dimension Belum Sinkron dengan Supabase

Blueprint Layer 4:

```text
e_p* = e_p^(0) || e_p^(1) || ... || e_p^(L)
```

Dengan konfigurasi:

```text
embed_size = 64
layer_size = [64,32,16]
```

Dimensi final:

```text
64 + 64 + 32 + 16 = 176
```

Sementara `04_Joint_Research_Azam_Raissa.md` sempat menulis:

```text
product_embeddings embedding VECTOR(64)
```

Ini mismatch.

Instruksi finalisasi:

```text
Pilih salah satu dan kunci untuk paper + Supabase:

Opsi A (direkomendasikan untuk menjaga formula KGAT/CR-HKGE):
  product_embeddings.embedding VECTOR(176)

Opsi B:
  Tambah projection layer dari 176 -> 64 dan jelaskan di paper.

Jangan diam-diam memotong embedding 176 menjadi 64.
```

Untuk hari ini, pilihan paling cepat dan paling konsisten dengan formula:

```text
Gunakan VECTOR(176).
```

### Gap 3: Cross-Reference Propagation Belum Strict Sesuai Rumus

Blueprint:

```text
e_CR(p) = lambda_ib * W_CR * (e_g + sum pi_CR(g,r',t') * e_t')
```

Status sekarang:

```text
CR-HKGE sudah mengambil global reference dan global attributes.
Namun global attribute context masih memakai row-normalized sparse propagation.
Belum strict memakai pi_CR(g,r',t') untuk setiap neighbor global reference.
```

Instruksi finalisasi:

```text
1. Buat subset triples global reference:
   g --has_global_accord--> t
   g --belongs_to_global_family--> t

2. Hitung score untuk setiap edge:
   score(g,r,t) = (W_r e_t)^T tanh(W_r e_g + e_r)

3. Terapkan relation-type weight:
   score_CR(g,r,t) = lambda_r * score(g,r,t)

4. Terapkan sparse softmax per global reference node g.

5. Bentuk:
   context_g = e_g + sum attention_CR(g,r,t) * e_t

6. Untuk produk p dengan inspired_by ke g:
   e_CR(p) = lambda_inspired_by * W_CR * context_g

7. Tambahkan e_CR(p) hanya untuk product rows enriched.

8. Produk tanpa inspired_by tetap mengikuti KGAT.
```

Target final:

```text
Cross-reference propagation harus benar-benar neighbor-attention based,
bukan hanya uniform/global-attribute averaging.
```

### Gap 4: Relation-Type Attention Perlu Dikunci Sesuai Formula Paper

Blueprint:

```text
pi_CR(h,r,t) = lambda_r * (W_r e_t)^T tanh(W_r e_h + e_r)
lambda_r = softmax(lambda_r)
```

Status sekarang:

```text
Relation-type scalar sudah ada.
Semantic relation tying sudah ada.
Relation weights sudah dipakai di attention score dan relation-aware propagation.
```

Risiko:

```text
Relation-aware propagation adalah tambahan implementasi yang lebih luas dari formula blueprint.
Jika dipakai sebagai bagian final, harus ditulis jelas di paper sebagai implementation detail.
Jika tidak ingin menambah novelty yang tidak tertulis, jadikan relation-aware propagation optional dan matikan pada final strict-formula run.
```

Instruksi finalisasi:

```text
1. Tambahkan flag:
   --cr_relation_aware_message 0/1

2. Untuk final strict-formula CR-HKGE:
   relation weights wajib memodifikasi pi_CR.
   cross-reference wajib memakai pi_CR.

3. Relation-aware propagation boleh:
   a. dimatikan agar formula paper bersih, atau
   b. dipertahankan jika paper menjelaskan bahwa e_N_int(p) dihitung dengan relation-type weighted message aggregation.

4. Jangan biarkan implementasi final memiliki mekanisme tambahan yang tidak dijelaskan di paper.
```

Rekomendasi pragmatis:

```text
Jika target hari ini adalah paper conference cepat:
  pertahankan relation-aware propagation hanya jika paper menulisnya sebagai part of relation-type weighted neighborhood aggregation.

Jika ingin formula paling bersih:
  tambahkan flag dan set --cr_relation_aware_message 0 untuk final strict run.
```

### Gap 5: KG Path Builder Belum Siap sebagai API Contract Final

`04_Joint_Research_Azam_Raissa.md` menyatakan `kg_path` adalah komponen kritikal untuk XAI Raissa.

Status sekarang:

```text
Artifact kg_paths.jsonl sudah dapat diexport.
Namun kg_path final untuk recommendation response harus dipilih berdasarkan query match, bukan sekadar semua triples produk.
```

Instruksi finalisasi:

```text
Untuk setiap product Top-3:

1. Ambil matched query entities.
2. Cari direct path:
   product --has_accord--> matched accord
   product --belongs_to_family--> matched family
   product --has_visual_note--> matched note

3. Jika product enriched:
   tambahkan product --inspired_by--> global_ref
   tambahkan global_ref --has_global_accord--> global accord yang relevan
   tambahkan global_ref --belongs_to_global_family--> global family jika relevan

4. Set matched=true hanya untuk path yang sesuai preferensi user.
5. Set matched=false untuk informative path seperti inspired_by yang menjelaskan semantic enrichment.
6. Return maksimal path yang meaningful, bukan dump semua triples.
```

Target JSON:

```json
{
  "rank": 1,
  "product_id": "5",
  "product_name": "Moz Art",
  "match_score": 89.4,
  "kg_path": [
    {"relation": "has_accord", "entity": "vanilla", "matched": true, "reason": "Sesuai preferensi vanilla"},
    {"relation": "belongs_to_family", "entity": "AMBER", "matched": true, "reason": "Sesuai preferensi aroma hangat"},
    {"relation": "inspired_by", "entity": "JPG Scandal Pour Homme", "matched": false, "reason": "Memberi konteks referensi global"}
  ]
}
```

### Gap 6: Artifact Export Perlu Memuat Query Encoder Config

Status sekarang:

```text
product_embeddings.tsv
entity_embeddings.tsv
relation_weights.tsv
kg_paths.jsonl
model_config.json
```

Instruksi finalisasi:

Tambahkan:

```text
query_encoder_config.json
```

Isi minimal:

```json
{
  "embedding_dim": 176,
  "entity_matching": {
    "accords": ["accord", "global_accord"],
    "family": ["family", "global_family"],
    "notes": ["note"]
  },
  "relation_weights_used": true,
  "score_function": "cosine",
  "top_k": 3
}
```

Tujuan:

```text
Serving layer di Supabase/Edge Function tahu bagaimana query vector dibentuk.
Paper bisa menjelaskan retrieval secara repeatable.
```

---

## 5. Finalization Plan Sebelum Ablation

User request:

```text
Tidak ingin ablation dulu kalau model CR-HKGE belum final.
```

Keputusan:

```text
Ablation ditunda.
Comparison model ditunda.
Prioritas sekarang adalah final model correctness terhadap novelty.
```

Urutan kerja yang harus dilakukan:

### Step 1: Lock Formula Implementation

Checklist:

```text
[ ] Layer 1 TransR tetap identik KGAT.
[ ] Layer 2 pi_CR memakai relation-type scalar lambda_r.
[ ] lambda_r diexport dan dapat dibaca per relation type.
[ ] Layer 3 e_CR(p) memakai global reference context.
[ ] e_CR(p) hanya masuk ke enriched products.
[ ] standard products tetap mengikuti KGAT propagation.
[ ] Tidak ada mekanisme tambahan yang tidak tertulis di paper, kecuali dijelaskan sebagai implementation detail.
```

### Step 2: Strict Cross-Reference Attention

Checklist:

```text
[ ] Ambil global reference triples.
[ ] Hitung edge attention pi_CR(g,r,t).
[ ] Sparse softmax per global reference node.
[ ] Bentuk context_g = e_g + weighted neighbor sum.
[ ] Inject context_g ke product p via inspired_by.
```

### Step 3: Query Encoder Final

Checklist:

```text
[ ] Input structured query dari NLU.
[ ] Entity matching ke KG.
[ ] Weighted entity aggregation.
[ ] Query vector dimension cocok dengan product embedding.
[ ] Cosine similarity ke 340 product embeddings.
[ ] Return Top-3.
```

### Step 4: KG Path Final

Checklist:

```text
[ ] kg_path dipilih berdasarkan matched query.
[ ] kg_path memuat relation, entity, matched, reason.
[ ] inspired_by path muncul untuk enriched product.
[ ] kg_path siap dipakai XAI Raissa.
```

### Step 5: Supabase Artifact Readiness

Checklist:

```text
[ ] product_embeddings dimensi final jelas: 176 atau projected 64.
[ ] products table sinkron dengan product2id.
[ ] kg_edges / kg_paths tersedia.
[ ] relation_weights tersedia.
[ ] model_config tersedia.
[ ] query_encoder_config tersedia.
[ ] model_version disimpan.
```

### Step 6: Final Smoke Test End-to-End

Gunakan contoh structured query manual sebelum masuk Edge Function:

```json
{
  "accords": ["vanilla", "sweet", "amber"],
  "family": "AMBER",
  "occasion": "evening"
}
```

Expected output:

```text
1. Sistem menghasilkan query vector.
2. Sistem mengembalikan Top-3 product.
3. Tiap product memiliki match_score.
4. Tiap product memiliki kg_path.
5. Enriched product memiliki inspired_by path.
6. Output JSON sesuai contract Raissa.
```

Jika ini belum berhasil, model belum final untuk paper system pipeline.

---

## 6. Commands yang Tetap Dipakai

### Pull Update di Colab

```bash
%cd /content/kgat
!git pull origin master
```

### Training CR-HKGE Saat Ini

```bash
%cd /content/kgat/Model

!python Main.py \
  --model_type cr_hkge \
  --data_path ../ \
  --dataset dataset-aromatique-kgat-ready \
  --alg_type bi \
  --adj_type si \
  --regs [1e-5,1e-5] \
  --layer_size [64,32,16] \
  --embed_size 64 \
  --kge_size 64 \
  --lr 0.0001 \
  --epoch 100 \
  --batch_size 64 \
  --batch_size_kg 256 \
  --mess_dropout [0.1,0.1,0.1] \
  --node_dropout [0.1] \
  --pretrain 0 \
  --save_flag 1 \
  --Ks [3,5,10] \
  --cr_use_relation_weight 1 \
  --cr_use_cross_ref 1 \
  --cr_relation_weight_mode semantic \
  --cr_export_embeddings 1 \
  --cr_artifact_path ../artifacts/cr_hkge \
  --cr_model_version cr_hkge_v1
```

### Export dari Checkpoint

Jika checkpoint sudah ada:

```bash
%cd /content/kgat/Model

!python Main.py \
  --model_type cr_hkge \
  --data_path ../ \
  --dataset dataset-aromatique-kgat-ready \
  --alg_type bi \
  --adj_type si \
  --regs [1e-5,1e-5] \
  --layer_size [64,32,16] \
  --embed_size 64 \
  --kge_size 64 \
  --lr 0.0001 \
  --epoch 0 \
  --batch_size 64 \
  --batch_size_kg 256 \
  --mess_dropout [0.1,0.1,0.1] \
  --node_dropout [0.1] \
  --pretrain 1 \
  --save_flag 0 \
  --Ks [3,5,10] \
  --cr_use_relation_weight 1 \
  --cr_use_cross_ref 1 \
  --cr_relation_weight_mode semantic \
  --cr_export_embeddings 1 \
  --cr_artifact_path ../artifacts/cr_hkge \
  --cr_model_version cr_hkge_v1
```

---

## 7. Paper Conference Readiness Criteria

CR-HKGE boleh disebut ready untuk paper conference jika semua ini terpenuhi:

```text
[ ] Paper menjelaskan problem 0 user interaction history sesuai blueprint.
[ ] Paper menjelaskan KGAT digunakan sebagai base model yang dimodifikasi.
[ ] Paper menjelaskan heterogeneous fragrance KG dengan 7 relation types.
[ ] Paper menjelaskan relation-type specific attention weights.
[ ] Paper menjelaskan cross-reference semantic propagation via inspired_by.
[ ] Implementasi cross-reference mengikuti rumus e_CR(p).
[ ] Query encoder structured query -> vector sudah ada.
[ ] Retrieval Top-3 product embeddings sudah ada.
[ ] KG paths untuk XAI sudah ada.
[ ] Output JSON sesuai kebutuhan Raissa.
[ ] Embedding dimension sinkron dengan Supabase schema.
[ ] Artifact export repeatable.
```

Yang belum perlu dilakukan sebelum model final:

```text
1. Ablation study.
2. Model comparison table final.
3. Multi-seed report.
4. Statistical significance testing.
```

Yang boleh dilakukan setelah model final:

```text
1. KGAT vanilla vs CR-HKGE pada Ks=[3,5,10].
2. BPRMF/CKE/NFM baseline comparison.
3. Ablation relation-weight only / cross-reference only.
4. Multi-seed mean +/- std.
5. Enriched vs standard product subset analysis.
```

---

## 8. Priority Hari Ini

Target hari ini:

```text
CR-HKGE final sesuai novelty, bukan ablation.
```

Prioritas paling penting:

```text
1. Finalisasi strict cross-reference attention sesuai e_CR(p).
2. Finalisasi query encoder structured query -> vector.
3. Finalisasi Top-3 retrieval + kg_paths.
4. Sinkronkan embedding dimension dengan Supabase.
5. Export artifact lengkap.
6. Jalankan end-to-end manual query test.
```

Jika waktu terbatas, minimum viable final untuk paper:

```text
1. Product embeddings CR-HKGE tersedia.
2. Query encoder tersedia.
3. Top-3 retrieval tersedia.
4. kg_path tersedia.
5. e_CR(p) sesuai rumus blueprint.
```

Baru setelah itu masuk:

```text
impact CR-HKGE dibanding model lain.
```
