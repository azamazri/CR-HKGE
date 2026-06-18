# IEEE-CRHKGE.md

## Brief Lengkap untuk Claude: Generate Paper Conference IEEE CR-HKGE Maksimal 5 Halaman A4

Dokumen ini adalah **brief kerja untuk Claude**, bukan isi paper final. Tujuannya adalah memberi Claude konteks lengkap agar dapat menulis paper conference IEEE maksimal 5 halaman A4 dengan struktur, klaim, rumus, gambar, tabel, hasil eksperimen, dan referensi yang konsisten dengan blueprint penelitian CR-HKGE.

Instruksi penting untuk Claude:

```text
Tulis paper final dalam bahasa Inggris akademik.
Format mengikuti gaya conference IEEE.
Maksimal 5 halaman A4, two-column IEEE style.
Jangan memasukkan pembahasan NFM sebagai baseline, tabel, grafik, atau discussion utama.
Gunakan maksimal 20 referensi.
Jangan mengarang sitasi. Jika metadata referensi tidak tersedia, beri tanda [VERIFY].
Jangan menyebut train.txt/test.txt sebagai historical user interaction.
Gunakan istilah: content-based positive pairs, surrogate supervision, zero historical interaction setting.
```

Model final yang digunakan:

```text
CR-HKGE final variant: cr_hkge_final_alpha_0_1
Cross-reference alpha: 0.1
Relation prior strength: 1.0
Dataset: dataset-aromatique-crhkge-ready
```

Baseline yang boleh dibahas:

```text
BPRMF
CKE
CFKG
KGAT
CR-HKGE
```

Baseline yang **tidak perlu dibahas dalam paper 5 halaman**:

```text
NFM
```

Alasan NFM tidak dimasukkan:

```text
Paper ini fokus pada knowledge graph recommendation dan modifikasi KGAT.
NFM adalah feature-interaction baseline, bukan KG reasoning baseline.
Untuk paper 5 halaman, pembahasan NFM akan menghabiskan ruang dan mengalihkan fokus dari novelty CR-HKGE.
```

---

## 1. Target Paper 5 Halaman

### 1.1 Distribusi halaman yang disarankan

Claude harus menjaga paper maksimal 5 halaman dengan alokasi:

| Bagian | Estimasi ruang |
|---|---:|
| Abstract + Keywords | 0.25 halaman |
| Introduction | 0.75 halaman |
| Related Work | 0.60 halaman |
| Proposed Method | 1.30 halaman |
| Experimental Setup | 0.65 halaman |
| Results and Discussion | 1.00 halaman |
| Conclusion | 0.25 halaman |
| References | 0.20-0.40 halaman |

Jika terlalu panjang, prioritas pemotongan:

```text
1. Kurangi Related Work.
2. Gabungkan Experimental Setup dan Results.
3. Gunakan 3 gambar maksimal.
4. Gunakan 4 tabel maksimal.
5. Jangan masukkan NFM.
```

### 1.2 Struktur final paper

Gunakan struktur:

```text
Abstract
Keywords
I. Introduction
II. Related Work
III. Proposed Method
IV. Experimental Setup
V. Results and Discussion
VI. Conclusion
References
```

Jika perlu lebih padat:

```text
III. Proposed CR-HKGE Method
IV. Experiments and Discussion
```

---

## 2. Core Message Paper

### 2.1 Masalah utama

Fragrance recommendation pada katalog lokal adalah masalah cold-start karena:

```text
1. Produk parfum hanya 340 item.
2. Tidak ada historical purchase, rating, atau click log.
3. Preferensi parfum bersifat semantik: accord, family, note, occasion, reference perfume.
4. Produk lokal memiliki atribut inspired_by yang menghubungkan produk lokal ke parfum global.
```

### 2.2 Gap penelitian

Paper harus menjelaskan gap berikut:

```text
KGAT dan banyak KG recommender biasanya mengasumsikan adanya user-item interaction matrix.
Aromatique tidak memiliki user interaction historis.
KGAT juga tidak memberi prioritas eksplisit pada tipe relasi domain-specific seperti inspired_by, has_accord, dan belongs_to_family.
Relasi inspired_by bukan atribut biasa; ia adalah semantic bridge dari local fragrance product ke global fragrance reference.
```

### 2.3 Solusi

Solusi yang ditawarkan:

```text
CR-HKGE: Cross-Reference Semantic Enrichment on Heterogeneous Knowledge Graph Embedding.
```

CR-HKGE memperluas KGAT dengan:

```text
1. Fragrance-specific heterogeneous KG construction.
2. Cross-reference semantic propagation through inspired_by.
3. Relation-type specific attention weights.
4. Content-based positive pairs for KGAT-compatible offline training without historical user behavior.
```

### 2.4 Klaim utama

Klaim aman:

```text
CR-HKGE outperforms KGAT, BPRMF, CKE, and CFKG on the CR-HKGE-ready Aromatique dataset.
```

Klaim enriched:

```text
CR-HKGE improves recommendation performance on enriched products connected to global references through inspired_by.
```

Klaim method:

```text
Relation-type attention and cross-reference propagation enable KGAT-style recommendation to better exploit fragrance-specific relational semantics.
```

Jangan menulis:

```text
CR-HKGE is better than all possible baselines.
CR-HKGE uses real user interaction data.
CR-HKGE proves online user satisfaction.
```

---

## 3. Abstract

### 3.1 Poin wajib abstract

Abstract harus memuat:

```text
1. Fragrance recommendation as cold-start / interaction-sparse problem.
2. Limitation of KGAT-style models requiring user-item interactions.
3. CR-HKGE as proposed model.
4. Three novelty components.
5. Content-based positive pairs as surrogate supervision.
6. Dataset statistics.
7. Results: CR-HKGE outperforms KGAT and KG/CF baselines.
```

### 3.2 Draft abstract untuk Claude

Claude boleh mengembangkan dari draft ini:

```text
Fragrance recommendation is a challenging cold-start problem because small-scale perfume catalogs often lack historical user-item interactions, while product relevance is expressed through subjective olfactory concepts such as accords, fragrance families, visual notes, and global fragrance references. Existing knowledge graph recommendation models, including KGAT, commonly rely on user-item interaction matrices and do not explicitly model cross-reference semantics between local perfume products and global fragrance references. This paper proposes CR-HKGE, a Cross-Reference Semantic Enrichment model on top of heterogeneous knowledge graph embedding for fragrance recommendation without historical user behavior. CR-HKGE constructs a fragrance-specific heterogeneous knowledge graph, propagates global reference semantics through the inspired_by relation, and applies relation-type attention to prioritize informative fragrance relations. Since the Aromatique dataset contains no purchase, rating, or click logs, content-based positive pairs are constructed as surrogate supervision for KGAT-compatible offline training and evaluation. Experiments on 340 perfume products, 998 entities, 9250 triples, and seven relation types show that CR-HKGE outperforms KGAT and other KG/CF baselines across overall Top-K metrics. These results indicate that cross-reference semantic enrichment can improve KG-based fragrance recommendation in zero historical interaction settings.
```

Catatan:

```text
Jangan menyebut NFM di abstract.
Jangan menulis "all baselines" jika baseline yang dibahas hanya BPRMF, CKE, CFKG, KGAT.
Gunakan "KG/CF baselines" atau sebutkan eksplisit.
```

---

## 4. Keywords

Gunakan 5-7 keyword:

```text
knowledge graph recommendation;
heterogeneous knowledge graph;
fragrance recommendation;
cold-start recommendation;
cross-reference enrichment;
relation-type attention;
KGAT
```

---

## 5. I. Introduction

### 5.1 Paragraf 1: Domain problem

Poin:

```text
Fragrance recommendation berbeda dari recommendation umum.
Preferensi user sering berupa deskripsi olfactory: sweet, woody, amber, vanilla, fresh.
User juga sering menyebut reference perfume atau aroma yang mirip dengan parfum populer.
Katalog lokal kecil dan tidak punya riwayat interaksi.
```

Kalimat contoh:

```text
Unlike movie or book recommendation, fragrance recommendation often relies on subjective olfactory descriptors and reference scents rather than explicit ratings or purchase histories.
```

### 5.2 Paragraf 2: Problem with collaborative filtering

Poin:

```text
Collaborative filtering butuh user-item interaction matrix.
Aromatique tidak punya purchase/rating/click logs.
Karena itu KGAT original tidak bisa diposisikan sebagai user-behavior model murni.
```

Kalimat contoh:

```text
In the Aromatique catalog, no historical purchase, rating, or click logs are available. This makes standard collaborative filtering and KGAT-style training difficult without constructing a surrogate supervision signal.
```

### 5.3 Paragraf 3: Why KG fits fragrance

Poin:

```text
Fragrance data naturally relational.
Product connected to accord, family, visual note, and global reference.
Global reference has its own accord/family.
Inspired_by connects local product with global semantic space.
```

### 5.4 Paragraf 4: Gap in KGAT and existing KG recommenders

Poin:

```text
KGAT uses attentive propagation but tidak eksplisit memberi bobot prior tiap relation type domain.
KGAT tidak punya cross-reference propagation khusus untuk inspired_by.
Existing KG recommender umumnya diuji pada dataset besar dengan user interactions.
```

### 5.5 Paragraf 5: Proposed CR-HKGE

Poin:

```text
CR-HKGE modifies KGAT for fragrance recommendation.
Uses content-based positive pairs.
Adds heterogeneous fragrance KG, cross-reference propagation, relation attention.
```

### 5.6 Contributions

Claude harus menulis kontribusi sebagai 4-5 bullet:

```text
1. A fragrance-specific heterogeneous KG is constructed for the Aromatique catalog, containing 340 products, 998 entities, 9250 triples, and seven relation types.

2. A cross-reference semantic propagation layer is introduced to transfer global fragrance reference semantics to local product embeddings through inspired_by.

3. Relation-type attention is introduced to differentiate the contribution of heterogeneous fragrance relations during KG propagation.

4. Content-based positive pairs are constructed as surrogate supervision for KGAT-compatible training in a zero historical interaction setting.

5. Experiments show that CR-HKGE improves over KGAT, BPRMF, CKE, and CFKG on overall Top-K recommendation metrics.
```

---

## 6. II. Related Work

### 6.1 Batasan related work untuk 5 halaman

Maksimal 3 subbagian:

```text
A. Collaborative and KG-Based Recommendation
B. Heterogeneous KG and Relation Attention
C. Cold-Start and Conversational Fragrance Recommendation
```

Jangan buat related work terlalu panjang. Maksimal 5-6 paragraf.

### 6.2 Referensi utama dari blueprint

Claude harus memakai paper utama dari `reference/03_Research_Azam_Blueprint.md`.

Core references yang sudah ada di blueprint:

```text
1. KGAT, Wang et al., KDD 2019.
2. HGKR, Zhang et al., Scientific Reports 2023. [VERIFY metadata]
3. ML-KDGATMoco, Rong et al., Scientific Reports 2024. [VERIFY metadata]
4. HN-DKG, Wan et al., Wiley IJIS 2024. [VERIFY metadata]
5. HKGAT, Zhang et al., Applied Intelligence 2025. [VERIFY metadata]
6. KGCN-UP, Liang et al., Scientific Reports 2025. [VERIFY metadata]
7. KGBPR, Ma et al., ICCPR 2022. [VERIFY metadata]
8. CKE, Zhang et al., KDD 2016.
```

Tambahan referensi yang disarankan agar total maksimal 20:

```text
9. BPR, Rendle et al., UAI 2009.
10. RippleNet, Wang et al., CIKM 2018.
11. KGCN, Wang et al., WWW 2019.
12. TransR, Lin et al., AAAI 2015.
13. Conversational recommender systems survey. [ADD verified survey]
14. Knowledge graph recommendation survey. [ADD verified survey]
15. Cold-start recommendation survey or paper. [ADD verified paper]
16. Heterogeneous information network recommendation paper. [ADD verified paper]
17. Graph neural network recommendation survey. [ADD verified survey]
18. Fragrance/perfume recommendation or olfactory descriptor paper. [ADD if verified]
19. One recent KG attention / relation-aware recommender paper. [ADD verified paper]
20. One domain-specific recommendation paper relevant to sparse catalog. [ADD verified paper]
```

Instruksi untuk Claude:

```text
Add up to 20 references total.
Do not invent paper titles, venues, years, DOIs, or authors.
If metadata is missing, mark [VERIFY] instead of fabricating.
Prefer peer-reviewed journals/conference papers from IEEE/ACM/Springer/Elsevier/Wiley/Nature Scientific Reports.
Do not include NFM as a discussed baseline reference because this 5-page version excludes NFM.
```

### 6.3 Related work content guide

Subsection A:

```text
Explain BPRMF, CKE, KGAT, CFKG as baselines.
Emphasize KGAT as direct architectural baseline.
```

Subsection B:

```text
Explain heterogeneous KG and relation-type weighting.
Connect to HN-DKG/HKGAT/HGKR limitations from blueprint.
```

Subsection C:

```text
Explain sparse/cold-start setting.
Connect to Aromatique: zero historical interaction.
Mention conversational recommendation only as system motivation, not main evaluation.
```

---

## 7. III. Proposed Method

### 7.1 Subsection A: Problem formulation

Define:

```text
P = set of perfume products
E = set of entities
R = set of relation types
T = set of triples
G = (E, R, T)
```

Triple:

```math
(h,r,t) \in T
```

Prediction:

```math
\hat{y}(u,p) = f(\mathbf{e}_u,\mathbf{e}_p)
```

Important wording:

```text
u denotes a content/query profile, not a real historical user.
```

### 7.2 Subsection B: Fragrance-specific heterogeneous KG construction

Explain node types:

```text
product
accord
note
family
global_ref
global_accord
global_family
```

Explain relation types:

| Relation | Meaning |
|---|---|
| `inspired_by` | local product -> global fragrance reference |
| `has_accord` | product -> local fragrance accord |
| `has_visual_note` | product -> visual note |
| `belongs_to_family` | product -> olfactory family |
| `sem_similar` | product -> semantically similar product |
| `has_global_accord` | global reference -> global accord |
| `belongs_to_global_family` | global reference -> global family |

Statistics:

| Component | Value |
|---|---:|
| Products | 340 |
| Entities | 998 |
| Relation types | 7 |
| KG triples | 9250 |
| Enriched products | 243 |
| Standard products | 97 |

Relation counts:

| Relation | Count |
|---|---:|
| `inspired_by` | 243 |
| `has_accord` | 1629 |
| `has_visual_note` | 680 |
| `belongs_to_family` | 340 |
| `sem_similar` | 4110 |
| `has_global_accord` | 2017 |
| `belongs_to_global_family` | 231 |

### 7.3 Subsection C: Content-based positive pair construction

Explain:

```text
Because Aromatique has no historical interactions, content-based positive pairs are generated.
Each content/query profile corresponds to a source product.
Positive targets are products with high relevance score based on local fragrance attributes and cross-reference semantics.
```

Dataset:

| Component | Value |
|---|---:|
| Content/query profiles | 340 |
| Train positive pairs | 2720 |
| Test positive pairs | 1360 |
| Train per profile | 8 |
| Test per profile | 4 |
| Unique train items | 329 |
| Unique test items | 301 |

Positive-pair scoring formula:

```math
s(p_i,p_j) =
w_a J(A_i,A_j)
+ w_n J(N_i,N_j)
+ w_f \mathbb{1}(F_i=F_j)
+ w_{ga} J(GA_i,GA_j)
+ w_{gf} J(GF_i,GF_j)
+ w_b B(i,j)
+ w_s \mathbb{1}(GR_i \cap GR_j \neq \emptyset)
+ w_{sim}\mathbb{1}(p_j \in S_i)
```

Definitions:

```text
A  = local accords
N  = visual notes
F  = olfactory family
GA = global accords
GF = global family
GR = global reference set
S  = semantic-similar neighbors
J  = Jaccard similarity
B  = global-to-local bridge score
```

Scoring weights:

| Component | Weight |
|---|---:|
| local accord | 2.00 |
| visual note | 1.25 |
| local family | 1.50 |
| global accord | 2.00 |
| global family | 1.25 |
| global-to-local accord bridge | 1.75 |
| global-to-local family bridge | 0.80 |
| same global reference | 3.00 |
| sem_similar | 0.35 |
| enriched cross-reference bonus | 0.20 |

### 7.4 Subsection D: CR-HKGE architecture

Claude should describe four layers:

```text
Layer 1: TransR Knowledge Graph Embedding
Layer 2: Relation-Type Specific Attention
Layer 3: Cross-Reference Semantic Propagation
Layer 4: Bi-Interaction Prediction
```

#### Layer 1: TransR KGE

Formula:

```math
g(h,r,t)=
\left\|
\mathbf{W}_r \mathbf{e}_h
+ \mathbf{e}_r
- \mathbf{W}_r \mathbf{e}_t
\right\|_2^2
```

KG loss:

```math
\mathcal{L}_{KG}
=
\sum_{(h,r,t,t')}
-\log \sigma(g(h,r,t')-g(h,r,t))
```

#### Layer 2: Relation-type attention

KGAT-style attention:

```math
\pi(h,r,t)=
(\mathbf{W}_r \mathbf{e}_t)^\top
\tanh(\mathbf{W}_r \mathbf{e}_h + \mathbf{e}_r)
```

CR-HKGE attention:

```math
\pi_{CR}(h,r,t)=
\tilde{\lambda}_r
\cdot
(\mathbf{W}_r \mathbf{e}_t)^\top
\tanh(\mathbf{W}_r \mathbf{e}_h + \mathbf{e}_r)
```

Relation normalization:

```math
\tilde{\lambda}_r =
\frac{\exp(\lambda_r)}
{\sum_{r'\in R}\exp(\lambda_{r'})}
```

#### Layer 3: Cross-reference propagation

Cross-reference context:

```math
\mathbf{c}_{ref}(p)=
\alpha \cdot \tilde{\lambda}_{ib}
\cdot \mathbf{W}_{ref}
\left(
\mathbf{e}_g+
\sum_{(g,r,t)\in N(g)}
\pi_{CR}(g,r,t)\mathbf{e}_t
\right)
```

Final alpha:

```text
alpha = 0.1
```

For standard products:

```math
\mathbf{c}_{ref}(p)=\mathbf{0}
```

Product update:

```math
\mathbf{e}^{(l)}_p =
\sigma\left(
\mathbf{W}^{(l)}
\left(
\mathbf{e}^{(l-1)}_p+
\mathbf{m}^{(l)}_{N(p)}+
\mathbf{c}_{ref}(p)
\right)
\right)
```

#### Layer 4: Prediction

Layer concatenation:

```math
\mathbf{e}^{*}_{p}
=
\mathbf{e}^{(0)}_p
\Vert
\mathbf{e}^{(1)}_p
\Vert
\dots
\Vert
\mathbf{e}^{(L)}_p
```

Prediction:

```math
\hat{y}(u,p)=
{\mathbf{e}^{*}_{u}}^\top
\mathbf{e}^{*}_{p}
```

### 7.5 Subsection E: Training objective

BPR loss:

```math
\mathcal{L}_{BPR}
=
-\sum_{(u,i,j)}
\log \sigma(\hat{y}(u,i)-\hat{y}(u,j))
```

Total loss:

```math
\mathcal{L}
=
\mathcal{L}_{BPR}
+\mathcal{L}_{KG}
+\beta \|\Theta\|_2^2
```

Training flow:

```text
Phase I: BPR training over content-based positive pairs.
Phase II: TransR/KG training and attentive adjacency update.
```

---

## 8. IV. Experimental Setup

### 8.1 Dataset

Use dataset table from Section 7.2 and 7.3.

Critical wording:

```text
The dataset contains no historical purchase, rating, or click logs.
The train/test files are content-based positive pairs used for offline surrogate evaluation.
```

### 8.2 Baselines

Only include:

| Model | Category | Purpose |
|---|---|---|
| BPRMF | Collaborative filtering | Basic ranking baseline |
| CKE | KG embedding | Collaborative KG embedding baseline |
| CFKG | KG collaborative baseline | KG-based collaborative baseline |
| KGAT | KG attention | Main architectural baseline |
| CR-HKGE | Proposed model | KGAT + fragrance KG + cross-reference + relation attention |

Do not include:

```text
NFM
```

### 8.3 Hyperparameters

| Parameter | Value |
|---|---:|
| Embedding size | 64 |
| KGE size | 64 |
| Layer size | [64, 32, 16] |
| Learning rate | 0.0001 |
| Batch size | 64 |
| Node dropout | [0.1] |
| Message dropout | [0.1, 0.1, 0.1] |
| Epochs | 100 |
| K values | [3, 5, 10] |
| CR-HKGE best metric | NDCG@3 |
| Final alpha | 0.1 |
| Relation prior strength | 1.0 |

### 8.4 Metrics

Precision@K:

```math
Precision@K = \frac{|Rel_K|}{K}
```

Recall@K:

```math
Recall@K = \frac{|Rel_K|}{|Rel|}
```

Hit@K:

```math
Hit@K = \mathbb{1}(|Rel_K|>0)
```

NDCG@K:

```math
DCG@K = \sum_{i=1}^{K}\frac{rel_i}{\log_2(i+1)}
```

```math
NDCG@K = \frac{DCG@K}{IDCG@K}
```

Optional if space allows:

```math
MRR =
\frac{1}{|Q|}
\sum_{q \in Q}
\frac{1}{rank_q}
```

If MRR/coverage is not in final logs, do not put it in the main results table.

---

## 9. V. Results and Discussion

### 9.1 Main overall result

Use this table in the paper.

| Model | Recall@3 | Precision@3 | Hit@3 | NDCG@3 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| CFKG | 0.00735 | 0.00980 | 0.02941 | 0.02027 | 0.03088 | 0.05142 |
| BPRMF | 0.21103 | 0.28137 | 0.65588 | 0.28089 | 0.53750 | 0.52198 |
| CKE | 0.22647 | 0.30196 | 0.68529 | 0.29865 | 0.57353 | 0.54575 |
| KGAT | 0.26324 | 0.35098 | 0.76176 | 0.32783 | 0.69118 | 0.60482 |
| CR-HKGE alpha=0.1 | 0.29485 | 0.39314 | 0.82647 | 0.36629 | 0.71397 | 0.62715 |

Discussion points:

```text
1. CR-HKGE outperforms all included baselines.
2. The improvement over KGAT demonstrates that the added cross-reference propagation and relation-type attention improve KGAT-style recommendation.
3. CFKG performs poorly, suggesting that simple collaborative KG modeling is insufficient in this small, interaction-free catalog.
4. BPRMF and CKE are stronger than CFKG but weaker than KGAT/CR-HKGE, indicating that attentive KG propagation is useful.
```

### 9.2 KGAT vs CR-HKGE gain

| Metric | KGAT | CR-HKGE | Gain |
|---|---:|---:|---:|
| Recall@3 | 0.26324 | 0.29485 | +0.03161 |
| Precision@3 | 0.35098 | 0.39314 | +0.04216 |
| Hit@3 | 0.76176 | 0.82647 | +0.06471 |
| NDCG@3 | 0.32783 | 0.36629 | +0.03846 |
| Recall@10 | 0.69118 | 0.71397 | +0.02279 |
| NDCG@10 | 0.60482 | 0.62715 | +0.02233 |

Discussion:

```text
The strongest improvement appears at Hit@3 and Precision@3, showing that CR-HKGE better places relevant candidates in the early recommendation list.
```

### 9.3 Enriched product analysis

| Model | Recall@3 | Precision@3 | Hit@3 | NDCG@3 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| KGAT | 0.25983 | 0.30285 | 0.68142 | 0.29933 | 0.69125 | 0.55936 |
| CR-HKGE alpha=0.1 | 0.30924 | 0.34808 | 0.76401 | 0.34737 | 0.72026 | 0.59119 |

Discussion:

```text
Enriched products are products with inspired_by relation.
CR-HKGE improves KGAT on enriched products, supporting the usefulness of cross-reference semantic propagation.
```

### 9.4 Standard product analysis

| Model | Recall@3 | Precision@3 | Hit@3 | NDCG@3 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| KGAT | 0.29206 | 0.15873 | 0.45714 | 0.22192 | 0.74286 | 0.45865 |
| CR-HKGE alpha=0.1 | 0.27302 | 0.14921 | 0.40952 | 0.20181 | 0.74127 | 0.43982 |

Discussion:

```text
KGAT remains slightly stronger on standard products, where inspired_by is absent.
This is expected because cross-reference propagation directly benefits enriched products.
However, CR-HKGE remains competitive on standard products and improves overall recommendation.
```

### 9.5 Alpha sensitivity

Use if space allows. If page limit is tight, move this to one short paragraph.

| Variant | Recall@3 | NDCG@3 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|
| alpha=0.25 | 0.29338 | 0.35838 | 0.71618 | 0.62163 |
| alpha=0.10 | 0.29485 | 0.36629 | 0.71397 | 0.62715 |
| alpha=0.075 | 0.28750 | 0.35847 | 0.71618 | 0.62510 |
| alpha=0.05 | 0.28382 | 0.35213 | 0.71912 | 0.62268 |

Discussion:

```text
Alpha=0.1 provides the best balance for overall early ranking and Top-10 performance.
Too large alpha can over-emphasize cross-reference paths.
Too small alpha weakens the contribution of global reference enrichment.
```

### 9.6 Ablation study

If time/page allows, use a compact ablation table.

Current available ablation table:

| Model | Recall@3 | Precision@3 | Hit@3 | NDCG@3 | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| A_no_novelty_modules | 0.26250 | 0.35000 | 0.74706 | 0.32442 | 0.70147 | 0.60681 |
| A_no_cross_reference | 0.28235 | 0.37647 | 0.79706 | 0.34955 | 0.71397 | 0.61842 |
| A_no_fragrance_prior | 0.28162 | 0.37549 | 0.76176 | 0.35621 | 0.69926 | 0.62106 |
| CR-HKGE alpha=0.1 | 0.29485 | 0.39314 | 0.82647 | 0.36629 | 0.71397 | 0.62715 |

Important caveat:

```text
If ablation is included, state that the final CR-HKGE variant uses alpha=0.1.
Avoid overclaiming relation attention if relation-attention ablation has mixed behavior in earlier logs.
```

---

## 10. Figures, Flowcharts, Charts, and Tables

### 10.1 Required figures

#### Figure 1. Overall Research Pipeline

Must show:

```text
Aromatique catalog
   -> Global reference enrichment
   -> Heterogeneous fragrance KG construction
   -> Content-based positive pair generation
   -> CR-HKGE training
   -> Top-K recommendation evaluation
```

Purpose:

```text
Clarifies that the model does not use real user interaction history.
```

#### Figure 2. Fragrance-Specific Heterogeneous KG Schema

Must show node types:

```text
product
accord
note
family
global_ref
global_accord
global_family
```

Must show relations:

```text
inspired_by
has_accord
has_visual_note
belongs_to_family
sem_similar
has_global_accord
belongs_to_global_family
```

Purpose:

```text
Supports Novelty 1.
```

#### Figure 3. CR-HKGE Architecture

Must show:

```text
TransR KGE
Relation-Type Attention
Cross-Reference Propagation
Bi-Interaction Prediction
```

Purpose:

```text
Supports Novelty 2 and Novelty 3.
```

### 10.2 Optional chart if page allows

#### Chart 1. Overall Baseline Comparison

Recommended:

```text
Bar chart of Recall@3, NDCG@3, Recall@10, NDCG@10 for KGAT and CR-HKGE.
```

If space is tight:

```text
Use table only, no chart.
```

#### Chart 2. Alpha Sensitivity

Recommended:

```text
Line chart with alpha on x-axis and NDCG@3 / Recall@10 on y-axis.
```

If space is tight:

```text
Use one sentence in Results instead of chart.
```

### 10.3 Required tables

Minimum tables:

```text
Table I: Dataset statistics
Table II: Main overall results
Table III: Enriched vs standard analysis
```

Optional if space allows:

```text
Table IV: Alpha sensitivity
Table V: Ablation study
```

---

## 11. VI. Conclusion

Conclusion must include:

```text
1. CR-HKGE adapts KGAT for fragrance recommendation without historical interactions.
2. Fragrance-specific heterogeneous KG captures local and global reference semantics.
3. Cross-reference propagation through inspired_by improves enriched product recommendation.
4. CR-HKGE outperforms KGAT, BPRMF, CKE, and CFKG.
5. Future work: real user evaluation, larger fragrance catalog, stronger relation calibration, online conversational recommendation.
```

Draft conclusion:

```text
This paper proposed CR-HKGE, a cross-reference semantic enrichment model for fragrance recommendation in a zero historical interaction setting. CR-HKGE constructs a fragrance-specific heterogeneous KG, propagates global reference semantics through the inspired_by relation, and applies relation-type attention to prioritize heterogeneous fragrance relations. Since Aromatique does not contain purchase, rating, or click logs, content-based positive pairs are constructed as surrogate supervision for KGAT-compatible training and evaluation. Experimental results show that CR-HKGE improves over KGAT and other KG/CF baselines across overall Top-K metrics. Future work will incorporate real user feedback, online conversational evaluation, larger fragrance catalogs, and more robust relation-type calibration.
```

---

## 12. Limitations and Future Work

Claude should include this briefly, either in Discussion or Conclusion:

Limitations:

```text
1. No real historical user interactions.
2. Positive pairs are surrogate content-based relevance labels.
3. Dataset has only 340 products.
4. Evaluation is offline, not online user testing.
5. Results need validation on other fragrance catalogs.
```

Future work:

```text
1. Collect real user feedback from conversational recommendation sessions.
2. Evaluate CR-HKGE in live chatbot recommendation flow.
3. Extend KG with season, occasion, gender, longevity, sillage, and price relations.
4. Improve relation-type attention calibration.
5. Test transferability to larger fragrance catalogs.
```

---

## 13. References Guide for Claude

### 13.1 Hard rule

```text
Use maximum 20 references.
Do not invent citations.
Use IEEE style.
If exact metadata is missing, write [VERIFY] and do not fabricate DOI/pages.
```

### 13.2 Must include references from blueprint

Claude should read `reference/03_Research_Azam_Blueprint.md` and include the relevant main references.

Core references:

```text
1. KGAT, Wang et al., KDD 2019.
2. CKE, Zhang et al., KDD 2016.
3. HGKR, Zhang et al., Scientific Reports 2023. [VERIFY]
4. ML-KDGATMoco, Rong et al., Scientific Reports 2024. [VERIFY]
5. HN-DKG, Wan et al., Wiley IJIS 2024. [VERIFY]
6. HKGAT, Zhang et al., Applied Intelligence 2025. [VERIFY]
7. KGCN-UP, Liang et al., Scientific Reports 2025. [VERIFY]
8. KGBPR, Ma et al., ICCPR 2022. [VERIFY]
```

### 13.3 Suggested additional references

Ask Claude to add verified references up to a maximum of 20 total:

```text
9. BPR, Rendle et al., UAI 2009.
10. TransR, Lin et al., AAAI 2015.
11. KGCN, Wang et al., WWW 2019.
12. RippleNet, Wang et al., CIKM 2018.
13. Knowledge graph recommendation survey. [VERIFY]
14. Conversational recommender systems survey. [VERIFY]
15. Cold-start recommendation survey/paper. [VERIFY]
16. Heterogeneous information network recommendation paper. [VERIFY]
17. Graph neural network recommendation survey. [VERIFY]
18. Domain-specific recommendation in sparse catalog setting. [VERIFY]
19. Relation-aware or edge-weighted KG recommendation paper. [VERIFY]
20. Fragrance/olfactory descriptor or perfume recommendation paper if available. [VERIFY]
```

Do not include NFM reference in this 5-page version.

---

## 14. Files to Attach to Claude

Attach only the following files. Do not attach every file in `reference/` because it will confuse the generation.

### 14.1 Required files

```text
IEEE-CRHKGE.md
reference/03_Research_Azam_Blueprint.md
reference/02_Dataset_Schema_Validation.md
reference/cr-hkge-bimbingan-implementasi-alpha-0-25.md
dataset-aromatique-crhkge-ready/summary.json
```

Why:

```text
IEEE-CRHKGE.md = paper generation brief.
03_Research_Azam_Blueprint.md = novelty, research gap, main references.
02_Dataset_Schema_Validation.md = dataset/KG validation.
cr-hkge-bimbingan...md = implementation explanation and final experiment narrative.
summary.json = exact dataset statistics.
```

### 14.2 Optional files if Claude needs implementation detail

```text
scripts/build_cr_hkge_ready_dataset.py
Model/CRHKGE.py
Model/Main.py
Model/evaluate_item_subsets.py
Model/utility/loader_kgat.py
```

Why:

```text
build_cr_hkge_ready_dataset.py = positive-pair construction and scoring formula.
CRHKGE.py = final model architecture.
Main.py = alternating training flow.
evaluate_item_subsets.py = evaluation protocol.
loader_kgat.py = KG metadata and CR-HKGE loader.
```

### 14.3 Optional dataset audit file

```text
dataset-aromatique-crhkge-ready/positive_pair_scores.tsv
```

Do not attach the full file if Claude has upload size limits. Instead, attach first 50-100 rows or summarize it.

### 14.4 Do not attach unless specifically needed

```text
reference/08_Store_Availability_Pilot_Test_Improvement_Guide.md
reference/06_XAI_ABC_GPT_NLU_Implementation_Guide.md
football_content_intelligence_os_blueprint.md
_external/
```

Reason:

```text
These files are not central to the IEEE CR-HKGE paper and may distract Claude from the CR-HKGE research scope.
```

---

## 15. Prompt to Give Claude

Use this prompt:

```text
You are helping write a maximum 5-page IEEE conference paper in English.

Use the attached IEEE-CRHKGE.md as the primary writing brief. Use the attached CR-HKGE blueprint and dataset validation files as supporting evidence. Generate a complete IEEE-style paper with:

1. Abstract
2. Keywords
3. Introduction
4. Related Work
5. Proposed Method
6. Experimental Setup
7. Results and Discussion
8. Conclusion
9. References

Hard constraints:
- Maximum 5 A4 pages in IEEE two-column style.
- Do not discuss NFM as a baseline.
- Do not claim that the dataset contains historical user interactions.
- Use "content-based positive pairs" and "surrogate supervision".
- Use maximum 20 references.
- Add relevant journal/conference references, but do not invent citation metadata. Mark uncertain metadata with [VERIFY].
- Preserve the three CR-HKGE novelty components:
  1. Fragrance-specific heterogeneous KG construction.
  2. Cross-reference semantic propagation via inspired_by.
  3. Relation-type specific attention.
- Include the provided formulas, but keep them concise.
- Include only the most important tables and figures due to 5-page limit.
- Report results only for BPRMF, CKE, CFKG, KGAT, and CR-HKGE.

Write the paper in polished academic English and make the claims conservative and defensible.
```

---

## 16. Final Reminder for Claude

Claude should prioritize:

```text
1. Correctness over overclaiming.
2. CR-HKGE vs KGAT as the central comparison.
3. Zero historical interaction framing.
4. Clear explanation of content-based positive pairs.
5. Compact but complete method section.
6. Maximum 5-page IEEE conference constraint.
```

