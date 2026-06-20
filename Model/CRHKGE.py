from utility.tf_compat import tf, xavier_initializer
from KGAT import KGAT
import json
import os
import time
import numpy as np


class CRHKGE(KGAT):
    """Implementasi utama CR-HKGE di atas KGAT.

    Alur besarnya:
    1. Tetap memakai pipeline KGAT: BPR untuk content-based positive pairs
       dan TransR/KGE untuk knowledge graph.
    2. Menambahkan relation-type attention agar tiap tipe relasi fragrance
       punya bobot semantik sendiri.
    3. Menambahkan cross-reference propagation untuk produk yang punya relasi
       `inspired_by` ke parfum global.
    """

    def _parse_args(self, data_config, pretrain_data, args):
        super(CRHKGE, self)._parse_args(data_config, pretrain_data, args)
        self.model_type = self.model_type.replace('kgat_', 'cr_hkge_', 1)

        # Fase konfigurasi CR-HKGE:
        # argumen ini mengaktifkan/mematikan novelty saat final training
        # maupun ablation study.
        self.cr_use_relation_weight = bool(int(getattr(args, 'cr_use_relation_weight', 1)))
        self.cr_use_cross_ref = bool(int(getattr(args, 'cr_use_cross_ref', 1)))
        self.cr_relation_weight_mode = getattr(args, 'cr_relation_weight_mode', 'semantic')
        self.cr_relation_prior_mode = getattr(args, 'cr_relation_prior_mode', 'none')
        self.cr_relation_prior_strength = float(getattr(args, 'cr_relation_prior_strength', 1.0))
        self.cr_relation_attention_scale = getattr(args, 'cr_relation_attention_scale', 'type_count')
        self.cr_relation_aware_message = bool(int(getattr(args, 'cr_relation_aware_message', 0)))
        # Plan B: gated conditional enrichment. When on, standard products get the
        # exact plain-KGAT update; only enriched products get the CR-HKGE treatment.
        self.cr_planB_gate = bool(int(getattr(args, 'cr_planB_gate', 1)))

        # Plan C+: residual enrichment (component 1) + discriminative contrastive
        # loss (component 2). Both leave KGAT message passing untouched and are
        # INDEPENDENTLY toggleable.
        self.cr_use_residual = bool(int(getattr(args, 'cr_use_residual', 1)))
        self.cr_residual_gamma_init = float(getattr(args, 'cr_residual_gamma', 0.1))
        self.cr_use_contrastive = bool(int(getattr(args, 'cr_use_contrastive', 1)))
        self.cr_contrastive_weight = float(getattr(args, 'cr_contrastive_weight', 0.1))
        self.cr_contrastive_margin = float(getattr(args, 'cr_contrastive_margin', 1.0))
        self.cr_contrastive_negs = int(getattr(args, 'cr_contrastive_negs', 5))
        self.cr_relation_message_scale = getattr(args, 'cr_relation_message_scale', 'type_count')
        self.cr_cross_ref_alpha = float(getattr(args, 'cr_cross_ref_alpha', 1.0))
        self.cr_cross_ref_bi_interaction = bool(int(getattr(args, 'cr_cross_ref_bi_interaction', 0)))
        self.cr_cross_ref_gate_enabled = bool(int(getattr(args, 'cr_cross_ref_gate', 0)))
        self.cr_cross_ref_gate_init = float(getattr(args, 'cr_cross_ref_gate_init', -2.0))
        self.cr_model_version = getattr(args, 'cr_model_version', 'cr_hkge_v1')

        # Metadata ini disiapkan oleh KGAT_loader khusus saat model_type=cr_hkge.
        # Isinya mencakup mapping relasi, produk enriched, dan matriks
        # product -> global reference untuk cross-reference propagation.
        self.cr_config = data_config.get('cr_hkge_config', {})
        self.cr_lap_list = data_config.get('lap_list', None)
        self.cr_adj_r_list = data_config.get('adj_r_list', None)
        self.cr_relation_type_ids = self._resolve_relation_type_ids()
        self.cr_relation_type_names = self._resolve_relation_type_names()
        self.cr_expanded_relation_names = self.cr_config.get(
            'expanded_relation_names',
            ['relation_%d' % i for i in range(self.n_relations)]
        )

        self.cr_inspired_expanded_relation_id = int(
            self.cr_config.get('inspired_by_expanded_relation_id', 1))
        self.cr_global_attr_relation_ids = [
            int(r) for r in self.cr_config.get('global_attr_relation_ids', [])
        ]

    def _resolve_relation_type_ids(self):
        # KGAT membuat relasi forward dan inverse sebagai relation id berbeda.
        # Pada CR-HKGE mode semantic, forward dan inverse dari relasi yang sama
        # dikembalikan ke satu tipe semantik agar bobot relasinya konsisten.
        if self.cr_relation_weight_mode == 'semantic':
            relation_type_ids = self.cr_config.get('expanded_relation_type_ids')
            if relation_type_ids is not None and len(relation_type_ids) == self.n_relations:
                return np.asarray(relation_type_ids, dtype=np.int32)

        return np.arange(self.n_relations, dtype=np.int32)

    def _resolve_relation_type_names(self):
        if self.cr_relation_weight_mode == 'semantic':
            relation_type_names = self.cr_config.get('relation_type_names')
            if relation_type_names:
                return list(relation_type_names)

        return self.cr_config.get(
            'expanded_relation_names',
            ['relation_%d' % i for i in range(self.n_relations)]
        )

    def _build_weights(self):
        all_weights = super(CRHKGE, self)._build_weights()
        initializer = xavier_initializer()

        self.cr_n_relation_types = len(self.cr_relation_type_names)

        if self.cr_use_relation_weight:
            # Novelty: relation-type specific attention.
            # Logit ini adalah parameter learnable lambda_r untuk tiap tipe
            # relasi fragrance. Setelah softmax, nilainya menjadi bobot relasi
            # yang mengalikan skor attention KGAT.
            initial_logits = self._initial_relation_type_logits()
            all_weights['cr_relation_type_logits'] = tf.Variable(
                tf.constant(initial_logits, dtype=tf.float32),
                name='cr_relation_type_logits')
            self.cr_relation_type_probs = tf.nn.softmax(
                all_weights['cr_relation_type_logits'],
                name='cr_relation_type_probs')
            if self.cr_relation_attention_scale == 'type_count':
                # type_count menjaga skala attention KGAT: jika bobot softmax
                # masih uniform, multiplier rata-rata menjadi 1, bukan 1/N.
                self.cr_relation_type_multipliers = (
                    self.cr_relation_type_probs * float(self.cr_n_relation_types))
            elif self.cr_relation_attention_scale == 'probability':
                self.cr_relation_type_multipliers = self.cr_relation_type_probs
            else:
                raise ValueError('unsupported cr_relation_attention_scale: %s' %
                                 self.cr_relation_attention_scale)

            if self.cr_relation_message_scale == 'type_count':
                self.cr_relation_type_message_multipliers = (
                    self.cr_relation_type_probs * float(self.cr_n_relation_types))
            elif self.cr_relation_message_scale == 'probability':
                self.cr_relation_type_message_multipliers = self.cr_relation_type_probs
            else:
                raise ValueError('unsupported cr_relation_message_scale: %s' %
                                 self.cr_relation_message_scale)
        else:
            self.cr_relation_type_probs = None
            self.cr_relation_type_multipliers = None
            self.cr_relation_type_message_multipliers = None

        # KGAT only sets self.weights after _build_weights returns. Both the legacy
        # cross-reference path and the Plan C+ residual need the relation embeddings
        # while building cross-reference tensors, so expose the dict early.
        self.weights = all_weights

        if self.cr_use_cross_ref:
            # Novelty: cross-reference propagation (legacy / Plan B per-layer path).
            for k in range(self.n_layers):
                current_dim = self.weight_size_list[k]
                all_weights['W_cr_%d' % k] = tf.Variable(
                    initializer([current_dim, current_dim]), name='W_cr_%d' % k)
                all_weights['b_cr_%d' % k] = tf.Variable(
                    initializer([1, current_dim]), name='b_cr_%d' % k)

            if self.cr_cross_ref_gate_enabled:
                all_weights['cr_cross_ref_gate_logit'] = tf.Variable(
                    tf.constant(self.cr_cross_ref_gate_init, dtype=tf.float32),
                    name='cr_cross_ref_gate_logit')
                self.cr_cross_ref_gate = tf.sigmoid(
                    all_weights['cr_cross_ref_gate_logit'],
                    name='cr_cross_ref_gate')
            else:
                self.cr_cross_ref_gate = tf.constant(1.0, dtype=tf.float32)
        else:
            self.cr_cross_ref_gate = tf.constant(0.0, dtype=tf.float32)

        # Plan C+ component 1: residual weights (final-embedding projection + gamma).
        # Built whenever the residual is on, independent of the legacy cross-ref.
        if self.cr_use_residual:
            final_dim = int(sum(self.weight_size_list))
            all_weights['W_cr_residual'] = tf.Variable(
                initializer([final_dim, final_dim]), name='W_cr_residual')
            all_weights['b_cr_residual'] = tf.Variable(
                initializer([1, final_dim]), name='b_cr_residual')
            all_weights['cr_residual_gamma'] = tf.Variable(
                tf.constant(self.cr_residual_gamma_init, dtype=tf.float32),
                name='cr_residual_gamma')
            self.cr_residual_gamma = all_weights['cr_residual_gamma']

        # Cross-reference TENSORS (product->global edges, masks, global-attr
        # attention) are needed by either the legacy cross-ref OR the residual.
        if self.cr_use_cross_ref or self.cr_use_residual:
            self._build_cross_ref_tensors()

        return all_weights

    def _initial_relation_type_logits(self):
        if self.cr_relation_prior_mode == 'none':
            return np.zeros([self.cr_n_relation_types], dtype=np.float32)

        if self.cr_relation_prior_mode != 'fragrance':
            raise ValueError('unsupported cr_relation_prior_mode: %s' %
                             self.cr_relation_prior_mode)

        # Prior domain fragrance:
        # nilai awal ini memberi sinyal bahwa relasi seperti sem_similar,
        # has_accord, belongs_to_family, dan inspired_by lebih informatif untuk
        # rekomendasi parfum. Nilai ini hanya inisialisasi; selama training
        # tetap dapat berubah karena logits-nya trainable.
        prior_by_name = {
            'interaction': 1.0,
            'inspired_by': 1.8,
            'has_accord': 2.0,
            'has_visual_note': 0.8,
            'belongs_to_family': 1.8,
            'sem_similar': 3.0,
            'has_global_accord': 1.2,
            'belongs_to_global_family': 1.2,
        }

        priors = []
        for relation_name in self.cr_relation_type_names:
            normalized_name = relation_name
            if normalized_name.startswith('inverse_'):
                normalized_name = normalized_name[len('inverse_'):]
            priors.append(prior_by_name.get(normalized_name, 1.0))

        priors = np.asarray(priors, dtype=np.float32)
        priors = np.maximum(priors, 1e-6)
        return np.log(priors) * self.cr_relation_prior_strength

    def _build_cross_ref_tensors(self):
        # Fase persiapan tensor cross-reference:
        # - product_global_mat: edge produk lokal -> parfum global reference
        # - product_mask: penanda produk enriched yang punya inspired_by
        # - global_attr_*: atribut milik parfum global seperti global accord
        #   dan global family.
        product_global_mat = self.cr_config.get('product_global_mat')
        product_mask = self.cr_config.get('product_mask')
        global_attr_relation_mats = self.cr_config.get('global_attr_relation_mats', [])
        global_attr_heads = np.asarray(
            self.cr_config.get('global_attr_attention_heads', []),
            dtype=np.int32)
        global_attr_relations = np.asarray(
            self.cr_config.get('global_attr_attention_relations', []),
            dtype=np.int32)
        global_attr_tails = np.asarray(
            self.cr_config.get('global_attr_attention_tails', []),
            dtype=np.int32)

        n_nodes = self.n_users + self.n_entities

        if product_global_mat is None:
            import scipy.sparse as sp
            product_global_mat = sp.coo_matrix((n_nodes, n_nodes), dtype=np.float32)

        if product_mask is None:
            product_mask = np.zeros((n_nodes, 1), dtype=np.float32)

        self.cr_product_global_tensor = self._convert_sp_mat_to_sp_tensor(product_global_mat)
        self.cr_product_mask_tensor = tf.constant(product_mask, dtype=tf.float32)
        self.cr_global_attr_relation_tensors = [
            (int(relation_id), self._convert_sp_mat_to_sp_tensor(mat))
            for relation_id, mat in global_attr_relation_mats
        ]
        self.cr_global_attr_attention_tensor = None
        self.cr_global_attr_attention_edge_count = int(len(global_attr_heads))

        if (len(global_attr_heads) > 0 and
                len(global_attr_heads) == len(global_attr_relations) == len(global_attr_tails)):
            self.cr_global_attr_attention_h = tf.constant(global_attr_heads, dtype=tf.int32)
            self.cr_global_attr_attention_r = tf.constant(global_attr_relations, dtype=tf.int32)
            self.cr_global_attr_attention_t = tf.constant(global_attr_tails, dtype=tf.int32)

            indices = np.column_stack((global_attr_heads, global_attr_tails)).astype(np.int64)
            self.cr_global_attr_attention_indices = tf.constant(indices, dtype=tf.int64)
            self.cr_global_attr_attention_shape = np.asarray([n_nodes, n_nodes], dtype=np.int64)
            self.cr_global_attr_attention_tensor = self._create_global_attr_attention_tensor()

    def _create_global_attr_attention_tensor(self):
        # Attention ini memilih atribut global reference yang paling relevan.
        # Skornya memakai formula TransR/KGAT yang sudah dimodifikasi dengan
        # relation-type multiplier CR-HKGE.
        scores = self._generate_transE_score(
            self.cr_global_attr_attention_h,
            self.cr_global_attr_attention_t,
            self.cr_global_attr_attention_r)

        attention_input = tf.SparseTensor(
            self.cr_global_attr_attention_indices,
            scores,
            self.cr_global_attr_attention_shape)
        return tf.sparse.softmax(attention_input)

    def _relation_multiplier_for_r(self, r):
        if not self.cr_use_relation_weight:
            return tf.ones_like(tf.cast(r, tf.float32), dtype=tf.float32)

        relation_type_ids = tf.constant(self.cr_relation_type_ids, dtype=tf.int32)
        selected_type_ids = tf.gather(relation_type_ids, r)
        return tf.gather(self.cr_relation_type_multipliers, selected_type_ids)

    def _relation_multiplier_for_expanded_id(self, expanded_relation_id):
        if not self.cr_use_relation_weight:
            return tf.constant(1.0, dtype=tf.float32)

        relation_type_id = int(self.cr_relation_type_ids[int(expanded_relation_id)])
        return tf.gather(self.cr_relation_type_multipliers, relation_type_id)

    def _relation_message_multiplier_for_expanded_id(self, expanded_relation_id):
        if not self.cr_use_relation_weight:
            return tf.constant(1.0, dtype=tf.float32)

        relation_type_id = int(self.cr_relation_type_ids[int(expanded_relation_id)])
        return tf.gather(self.cr_relation_type_message_multipliers, relation_type_id)

    def _relation_multiplier_for_raw_id(self, raw_relation_id):
        expanded_relation_id = int(raw_relation_id) + 1
        return self._relation_multiplier_for_expanded_id(expanded_relation_id)

    def _generate_transE_score(self, h, t, r):
        # Fase Layer 1 + Layer 2:
        # skor attention KGAT berbasis TransR dihitung, lalu dikalikan
        # multiplier relation-type attention CR-HKGE.
        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        embeddings = tf.expand_dims(embeddings, 1)

        h_e = tf.nn.embedding_lookup(embeddings, h)
        t_e = tf.nn.embedding_lookup(embeddings, t)

        r_e = tf.nn.embedding_lookup(self.weights['relation_embed'], r)
        trans_M = tf.nn.embedding_lookup(self.weights['trans_W'], r)

        h_e = tf.reshape(tf.matmul(h_e, trans_M), [-1, self.kge_dim])
        t_e = tf.reshape(tf.matmul(t_e, trans_M), [-1, self.kge_dim])

        kg_score = tf.reduce_sum(tf.multiply(t_e, tf.tanh(h_e + r_e)), 1)
        relation_multiplier = self._relation_multiplier_for_r(r)
        return kg_score * relation_multiplier

    def _build_kgat_attention_tensor(self):
        # Plan B: pure-KGAT attentive adjacency, computed IN-GRAPH and therefore
        # differentiable. This is the SAME attention plain KGAT uses -- it calls
        # the BASE KGAT TransR score (KGAT._generate_transE_score), i.e. WITHOUT the
        # CR-HKGE relation-type multiplier lambda_r. Standard products aggregate
        # over this so their neighborhood is never reweighted by lambda_r.
        cached = getattr(self, '_cr_kgat_A_tensor', None)
        if cached is not None:
            return cached

        n_nodes = self.n_users + self.n_entities
        h = tf.constant(np.asarray(self.all_h_list, dtype=np.int32))
        t = tf.constant(np.asarray(self.all_t_list, dtype=np.int32))
        r = tf.constant(np.asarray(self.all_r_list, dtype=np.int32))

        # Explicitly call the BASE-class score (no lambda_r). CRHKGE overrides
        # _generate_transE_score to multiply by the relation multiplier; we bypass
        # that here so the standard branch is bit-for-bit plain KGAT.
        pure_scores = KGAT._generate_transE_score(self, h=h, t=t, r=r)

        indices = np.column_stack((self.all_h_list, self.all_t_list)).astype(np.int64)
        sp_in = tf.SparseTensor(indices, pure_scores, [n_nodes, n_nodes])
        self._cr_kgat_A_tensor = tf.sparse.softmax(sp_in)
        return self._cr_kgat_A_tensor

    def _standard_product_gate(self):
        # Plan B gate g_std: 1.0 for STANDARD product nodes (item, no inspired_by),
        # 0.0 for everything else (enriched products, users, non-product entities).
        # Used as: side = side_CR + g_std * (side_KGAT - side_CR), so only standard
        # products are switched to the plain-KGAT message; enriched products / users
        # / entities keep the exact current CR-HKGE behaviour.
        cached = getattr(self, '_cr_standard_gate_tensor', None)
        if cached is not None:
            return cached

        n_nodes = self.n_users + self.n_entities
        if not self.cr_planB_gate:
            # Plan B disabled -> no node is forced to KGAT (legacy CR-HKGE).
            self._cr_standard_gate_tensor = tf.constant(
                np.zeros((n_nodes, 1), dtype=np.float32), dtype=tf.float32)
            self._cr_standard_gate_np = np.zeros((n_nodes, 1), dtype=np.float32)
            return self._cr_standard_gate_tensor
        # Mask of product (item) nodes.
        item_mask = np.zeros((n_nodes, 1), dtype=np.float32)
        item_mask[self.n_users:self.n_users + self.n_items, 0] = 1.0
        # Enriched product mask (1 for products that have inspired_by).
        product_mask = self.cr_config.get('product_mask')
        if product_mask is None:
            product_mask = np.zeros((n_nodes, 1), dtype=np.float32)
        product_mask = np.asarray(product_mask, dtype=np.float32).reshape((n_nodes, 1))
        # standard product = item node AND NOT enriched.
        standard_mask = item_mask * (1.0 - product_mask)
        self._cr_standard_gate_tensor = tf.constant(standard_mask, dtype=tf.float32)
        self._cr_standard_gate_np = standard_mask
        return self._cr_standard_gate_tensor

    def _kgat_reference_layer(self, ego_embeddings, kgat_A, k):
        # One plain-KGAT bi-interaction layer using the pure attention. Used both as
        # the standard-product branch and to materialise the invariant-check tensor.
        side_embeddings = tf.sparse_tensor_dense_matmul(kgat_A, ego_embeddings)

        add_embeddings = ego_embeddings + side_embeddings
        sum_embeddings = tf.nn.leaky_relu(
            tf.matmul(add_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

        bi_embeddings = tf.multiply(ego_embeddings, side_embeddings)
        bi_embeddings = tf.nn.leaky_relu(
            tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

        ego_next = bi_embeddings + sum_embeddings
        ego_next = tf.nn.dropout(ego_next, 1 - self.mess_dropout[k])
        return ego_next

    def _create_bi_interaction_embed(self):
        # Plan C+ wrapper: the message-passing core is UNCHANGED KGAT (when the
        # legacy novelties are off, which is the Plan C+ configuration), then a
        # residual is added ONLY at the final embedding. Capturing the pre-residual
        # (pure-KGAT) embeddings lets the invariants assert standard==KGAT and the
        # gamma=0 floor.
        ua_kgat, ea_kgat = self._bi_interaction_core()
        self.cr_dbg_ua_kgat = ua_kgat
        self.cr_dbg_ea_kgat = ea_kgat
        return self._apply_residual(ua_kgat, ea_kgat)

    def _bi_interaction_core(self, gate_tensor=None):
        if not self.cr_use_cross_ref and not self.cr_relation_aware_message:
            return super(CRHKGE, self)._create_bi_interaction_embed()

        # Fase Layer 4 (Plan B: gated conditional enrichment):
        # fungsi ini menggantikan bi-interaction KGAT ketika novelty CR-HKGE aktif.
        # Untuk produk STANDARD (tanpa inspired_by), gate memaksa update menjadi
        # identik dengan KGAT murni; produk ENRICHED tetap mendapat treatment CR-HKGE
        # penuh. e_p = g_p * CR_update(p) + (1 - g_p) * KGAT_update(p), dengan
        # g_p = 1 untuk enriched dan 0 untuk standard.
        #
        # gate_tensor: None -> primary (gated) stream using the standard-product gate.
        # A zero gate reproduces the ORIGINAL pre-Plan-B CR-HKGE forward exactly
        # (used only by the diagnostic to compare enriched products).
        primary = gate_tensor is None

        A = self.A_in
        A_fold_hat = self._split_A_hat(A)
        relation_A_fold_hat = self._build_relation_aware_A_fold_hat()

        # Plan B in-graph machinery.
        kgat_A = self._build_kgat_attention_tensor()
        standard_gate = self._standard_product_gate() if primary else gate_tensor

        ego_embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            if relation_A_fold_hat is not None:
                # CR side: relation-aware message (deviation c) + lambda_r weights.
                side_cr = self._relation_aware_side_embeddings(
                    ego_embeddings,
                    relation_A_fold_hat)
            else:
                # CR side fallback: lambda_r-weighted attentive adjacency A_in.
                temp_embed = []
                for f in range(self.n_fold):
                    temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))
                side_cr = tf.concat(temp_embed, 0)

            # KGAT side: pure attention (no lambda_r, no relation-aware message).
            side_kgat = tf.sparse_tensor_dense_matmul(kgat_A, ego_embeddings)

            # GATE the side message: standard products -> side_kgat, others -> side_cr.
            # This neutralises BOTH deviation (b) lambda_r-on-attention and (c)
            # relation-aware message for standard products in one differentiable op.
            side_embeddings = side_cr + standard_gate * (side_kgat - side_cr)

            sum_side_embeddings = side_embeddings
            bi_side_embeddings = side_embeddings
            if self.cr_use_cross_ref:
                # Deviation (a): cross-reference context. Already masked to enriched
                # products via cr_product_mask_tensor; we additionally multiply by
                # (1 - standard_gate) so standard products are *provably* unaffected.
                cross_ref_context = self._create_cross_reference_context(ego_embeddings, k)
                cross_ref_context = cross_ref_context * (1.0 - standard_gate)
                sum_side_embeddings = side_embeddings + cross_ref_context
                if self.cr_cross_ref_bi_interaction:
                    bi_side_embeddings = sum_side_embeddings

            add_embeddings = ego_embeddings + sum_side_embeddings

            sum_embeddings = tf.nn.leaky_relu(
                tf.matmul(add_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            bi_embeddings = tf.multiply(ego_embeddings, bi_side_embeddings)
            bi_embeddings = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            ego_next = bi_embeddings + sum_embeddings
            ego_next = tf.nn.dropout(ego_next, 1 - self.mess_dropout[k])

            if k == 0:
                # Invariant capture (layer 1, identical inputs for both streams):
                # the gated update vs the pure-KGAT update from the SAME ego_0.
                if primary:
                    self.cr_dbg_gated_layer1 = ego_next
                    self.cr_dbg_kgat_layer1 = self._kgat_reference_layer(
                        all_embeddings[0], kgat_A, 0)
                else:
                    # Ungated (original CR-HKGE) layer-1, for the enriched diagnostic.
                    self.cr_dbg_orig_layer1 = ego_next

            ego_embeddings = ego_next
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        ua_embeddings, ea_embeddings = tf.split(all_embeddings, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _create_residual_cross_reference_context(self, node_embeddings):
        # Plan C+ component 1. Reproduces the cross-reference context construction of
        # `_create_cross_reference_context` (global-attribute attention -> global
        # reference -> product, via inspired_by) but ONCE at the FINAL embedding
        # dimension, with its own projection W_cr_residual. No lambda_r / alpha /
        # gate confounds (gamma scales it outside). Masked to enriched products, so
        # standard products and users/entities receive a zero residual.
        if self.cr_global_attr_attention_tensor is not None:
            attr_context = tf.sparse_tensor_dense_matmul(
                self.cr_global_attr_attention_tensor, node_embeddings)
        else:
            attr_context = tf.zeros_like(node_embeddings)
            for _raw_relation_id, relation_tensor in self.cr_global_attr_relation_tensors:
                attr_context = attr_context + tf.sparse_tensor_dense_matmul(
                    relation_tensor, node_embeddings)

        global_reference_context = node_embeddings + attr_context
        product_context = tf.sparse_tensor_dense_matmul(
            self.cr_product_global_tensor, global_reference_context)

        transformed_context = tf.nn.leaky_relu(
            tf.matmul(product_context, self.weights['W_cr_residual']) +
            self.weights['b_cr_residual'])

        # g_p mask: nonzero only for enriched product nodes.
        return transformed_context * self.cr_product_mask_tensor

    def _apply_residual(self, ua_embeddings, ea_embeddings):
        # e_p_final = e_p_kgat + g_p * gamma * c_ref(p), applied once at the final
        # embedding. Safety net (Task 3): any failure here leaves the pure-KGAT
        # embeddings untouched so the contrastive component can still run.
        if not self.cr_use_residual:
            return ua_embeddings, ea_embeddings
        try:
            full = tf.concat([ua_embeddings, ea_embeddings], axis=0)
            c_ref = self._create_residual_cross_reference_context(full)  # already * g_p
            full = full + self.cr_residual_gamma * c_ref
            ua_out, ea_out = tf.split(full, [self.n_users, self.n_entities], 0)
            self.cr_residual_active = True
            return ua_out, ea_out
        except Exception as exc:  # pragma: no cover - defensive toggle isolation
            print('[Plan C+] residual disabled at build time due to: %r' % exc)
            self.cr_residual_active = False
            return ua_embeddings, ea_embeddings

    def _build_contrastive_pairs(self):
        # Plan C+ component 2 -- HARD-NEGATIVE sampling (ANTI-CIRCULAR).
        # For each enriched anchor a, pick the K products n that are MOST
        # attribute-similar (share accords/family) BUT have a DIFFERENT global
        # reference (disjoint inspired_by target). We push a and n APART. We never
        # add a same-reference positive (that would copy the test label) -- see
        # IMPLEMENTATION_NOTES_PLANCPLUS.md for the non-circularity argument.
        def _as_set(mapping, key):
            value = mapping.get(key, mapping.get(str(key), []))
            return set(int(v) for v in value)

        accords = self.cr_config.get('product_accords', {})
        families = self.cr_config.get('product_families', {})
        refs = self.cr_config.get('product_to_global_ref', {})
        enriched = sorted(int(p) for p in self.cr_config.get('enriched_product_ids', []))

        anchors, negatives = [], []
        for anchor in enriched:
            anchor_attr = _as_set(accords, anchor) | _as_set(families, anchor)
            anchor_ref = _as_set(refs, anchor)
            if not anchor_attr or not anchor_ref:
                continue
            candidates = []
            for other in range(self.n_items):
                if other == anchor:
                    continue
                other_ref = _as_set(refs, other)
                if anchor_ref & other_ref:
                    continue  # same global reference -> NOT a hard negative
                other_attr = _as_set(accords, other) | _as_set(families, other)
                overlap = anchor_attr & other_attr
                if overlap:
                    jaccard = len(overlap) / len(anchor_attr | other_attr)
                    candidates.append((jaccard, other))
            candidates.sort(key=lambda row: (-row[0], row[1]))
            for _jaccard, other in candidates[:self.cr_contrastive_negs]:
                anchors.append(anchor)
                negatives.append(other)

        self.cr_contrastive_anchor_ids = np.asarray(anchors, dtype=np.int32)
        self.cr_contrastive_neg_ids = np.asarray(negatives, dtype=np.int32)
        return self.cr_contrastive_anchor_ids, self.cr_contrastive_neg_ids

    def _build_contrastive_loss(self):
        # Margin-based hinge repulsion over (anchor, hard-negative) pairs:
        #   L = mean relu(margin - ||e_a - e_n||_2)
        # Pushes attribute-similar-but-reference-different products at least
        # `margin` apart. Returns 0 if disabled or no enriched anchors exist
        # (e.g. on a hold-out KG with inspired_by removed).
        anchor_ids, neg_ids = self._build_contrastive_pairs()
        self.cr_n_contrastive_pairs = int(len(anchor_ids))
        if len(anchor_ids) == 0:
            return tf.constant(0.0, dtype=tf.float32)

        anchor_e = tf.gather(self.ea_embeddings, tf.constant(anchor_ids, dtype=tf.int32))
        neg_e = tf.gather(self.ea_embeddings, tf.constant(neg_ids, dtype=tf.int32))
        dist = tf.sqrt(tf.reduce_sum(tf.square(anchor_e - neg_e), axis=1) + 1e-12)
        return tf.reduce_mean(tf.nn.relu(self.cr_contrastive_margin - dist))

    def _build_loss_phase_I(self):
        # Reuse KGAT's BPR + KG + reg loss EXACTLY, then add the discriminative
        # contrastive term and rebuild the optimizer over the combined loss.
        super(CRHKGE, self)._build_loss_phase_I()

        self.cr_contrastive_loss = tf.constant(0.0, dtype=tf.float32)
        self.cr_n_contrastive_pairs = 0
        if self.cr_use_contrastive:
            try:
                self.cr_contrastive_loss = self._build_contrastive_loss()
                self.loss = self.loss + self.cr_contrastive_weight * self.cr_contrastive_loss
                # Rebuild the optimizer over L_BPR + L_KG + lambda_c * L_disc.
                self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
                print('[Plan C+] contrastive loss active: %d hard-negative pairs, '
                      'lambda_c=%.3f, margin=%.2f' % (
                          self.cr_n_contrastive_pairs, self.cr_contrastive_weight,
                          self.cr_contrastive_margin))
            except Exception as exc:  # pragma: no cover - defensive toggle isolation
                print('[Plan C+] contrastive disabled at build time due to: %r' % exc)
                self.cr_contrastive_loss = tf.constant(0.0, dtype=tf.float32)

    def _build_relation_aware_A_fold_hat(self):
        # Menyiapkan adjacency per relasi untuk relation-aware message.
        # Jika dimatikan dalam ablation, fungsi ini mengembalikan None dan model
        # kembali ke message passing KGAT standar.
        if not self.cr_use_relation_weight:
            return None

        if not self.cr_relation_aware_message:
            return None

        if self.cr_lap_list is None or self.cr_adj_r_list is None:
            return None

        return [
            (int(relation_id), self._split_A_hat(lap.tocsr()))
            for relation_id, lap in zip(self.cr_adj_r_list, self.cr_lap_list)
        ]

    def _relation_aware_side_embeddings(self, ego_embeddings, relation_A_fold_hat):
        # Setiap relation-specific adjacency menghasilkan pesan sendiri.
        # Pesan tersebut dikalikan multiplier lambda_r agar relasi penting
        # memberi kontribusi lebih besar pada embedding produk.
        relation_messages = []

        for relation_id, A_fold_hat in relation_A_fold_hat:
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], ego_embeddings))

            relation_embedding = tf.concat(temp_embed, 0)
            relation_multiplier = self._relation_message_multiplier_for_expanded_id(relation_id)
            relation_messages.append(relation_embedding * relation_multiplier)

        return tf.add_n(relation_messages)

    def _create_cross_reference_context(self, ego_embeddings, layer_id):
        # Inti Novelty cross-reference:
        # 1. Ambil konteks atribut dari global reference.
        # 2. Alirkan konteks global reference ke produk lokal via inspired_by.
        # 3. Transformasi dengan W_cr agar dimensinya sesuai layer KGAT.
        # 4. Mask memastikan hanya produk enriched yang menerima konteks ini.
        if self.cr_global_attr_attention_tensor is not None:
            attr_context = tf.sparse_tensor_dense_matmul(
                self.cr_global_attr_attention_tensor,
                ego_embeddings)
        else:
            attr_context = tf.zeros_like(ego_embeddings)
            for raw_relation_id, relation_tensor in self.cr_global_attr_relation_tensors:
                relation_context = tf.sparse_tensor_dense_matmul(relation_tensor, ego_embeddings)
                relation_multiplier = self._relation_multiplier_for_raw_id(raw_relation_id)
                attr_context = attr_context + relation_context * relation_multiplier

        global_reference_context = ego_embeddings + attr_context
        product_context = tf.sparse_tensor_dense_matmul(
            self.cr_product_global_tensor,
            global_reference_context)

        transformed_context = tf.nn.leaky_relu(
            tf.matmul(product_context, self.weights['W_cr_%d' % layer_id]) +
            self.weights['b_cr_%d' % layer_id])

        inspired_multiplier = self._relation_multiplier_for_expanded_id(
            self.cr_inspired_expanded_relation_id)

        return (self.cr_cross_ref_alpha * self.cr_cross_ref_gate * inspired_multiplier *
                transformed_context * self.cr_product_mask_tensor)

    def check_standard_equals_kgat(self, sess, tol=1e-5, verbose=True):
        """Plan B invariant check.

        For STANDARD products (item nodes without inspired_by), the layer-1 gated
        CR-HKGE update must equal the plain-KGAT update bit-for-bit (within fp
        tolerance), given identical layer-0 inputs. Enriched products must differ
        (sanity that the CR path is actually doing something). Returns a dict and
        raises AssertionError if the invariant is violated.

        Run with message dropout = 0 so dropout is the identity.
        """
        if not hasattr(self, 'cr_dbg_gated_layer1'):
            raise RuntimeError(
                'invariant tensors not built; this check requires the gated '
                'CR-HKGE path (cr_use_cross_ref or cr_relation_aware_message on).')

        feed = {
            self.mess_dropout: [0.] * self.n_layers,
            self.node_dropout: [0.] * self.n_layers,
        }
        gated, kgat_ref = sess.run(
            [self.cr_dbg_gated_layer1, self.cr_dbg_kgat_layer1], feed_dict=feed)

        enriched_items = set(int(i) for i in self.cr_config.get('enriched_product_ids', []))
        standard_nodes = []
        enriched_nodes = []
        for item_id in range(self.n_items):
            node = self.n_users + item_id
            if item_id in enriched_items:
                enriched_nodes.append(node)
            else:
                standard_nodes.append(node)

        std_diff = 0.0
        if standard_nodes:
            std_diff = float(np.max(np.abs(gated[standard_nodes] - kgat_ref[standard_nodes])))
        enr_diff = 0.0
        if enriched_nodes:
            enr_diff = float(np.max(np.abs(gated[enriched_nodes] - kgat_ref[enriched_nodes])))

        result = {
            'n_standard_products': len(standard_nodes),
            'n_enriched_products': len(enriched_nodes),
            'max_abs_diff_standard_vs_kgat': std_diff,
            'max_abs_diff_enriched_vs_kgat': enr_diff,
            'tolerance': tol,
        }
        if verbose:
            print('[Plan B invariant] standard products: %d, enriched products: %d'
                  % (len(standard_nodes), len(enriched_nodes)))
            print('[Plan B invariant] max|gated - KGAT| on STANDARD = %.3e (must be <= %.1e)'
                  % (std_diff, tol))
            print('[Plan B invariant] max|gated - KGAT| on ENRICHED = %.3e (should be > 0)'
                  % enr_diff)

        assert std_diff <= tol, (
            'Plan B invariant VIOLATED: standard products deviate from KGAT by %.3e > %.1e'
            % (std_diff, tol))
        return result

    def build_planB_enriched_diagnostic(self):
        """Build a parallel UNGATED (== original CR-HKGE) stream sharing the SAME
        weights as the gated model, so we can compare enriched products
        gated-vs-original both at layer 1 and at the full multi-layer output.

        Must be called once after construction (adds graph tensors only; does not
        touch training).
        """
        n_nodes = self.n_users + self.n_entities
        zero_gate = tf.zeros((n_nodes, 1), dtype=tf.float32)
        # gate_tensor=zero_gate -> side = side_cr for every node, cross_ref unmasked
        # by the gate == the original pre-Plan-B CR-HKGE forward.
        _ua_orig, ea_orig = self._bi_interaction_core(gate_tensor=zero_gate)
        self.cr_dbg_full_orig_ea = ea_orig
        self.cr_dbg_full_gated_ea = self.ea_embeddings
        return self

    def check_enriched_equals_original(self, sess, tol=1e-5, verbose=True):
        """Plan B enriched invariant.

        Verifies whether, for ENRICHED products (cr_product_mask=1), the gated
        CR-HKGE-B output equals the ORIGINAL CR-HKGE output. Reports two levels:
          - layer 1 (identical inputs): expected to PASS by construction.
          - full multi-layer entity embedding: this is what feeds predictions.
        Returns a dict; does NOT raise (diagnostic only).
        """
        if not hasattr(self, 'cr_dbg_full_orig_ea'):
            self.build_planB_enriched_diagnostic()

        feed = {
            self.mess_dropout: [0.] * self.n_layers,
            self.node_dropout: [0.] * self.n_layers,
        }
        gated_l1, orig_l1, gated_ea, orig_ea = sess.run(
            [self.cr_dbg_gated_layer1, self.cr_dbg_orig_layer1,
             self.cr_dbg_full_gated_ea, self.cr_dbg_full_orig_ea],
            feed_dict=feed)

        enriched_items = sorted(int(i) for i in self.cr_config.get('enriched_product_ids', []))
        enriched_nodes = [self.n_users + i for i in enriched_items]

        # Layer 1: compare full-node tensors on enriched product NODES.
        l1_diff = 0.0
        if enriched_nodes:
            l1_diff = float(np.max(np.abs(gated_l1[enriched_nodes] - orig_l1[enriched_nodes])))
        # Full: compare entity embeddings on enriched product item rows.
        full_diff = 0.0
        per_layer_note = ''
        if enriched_items:
            full_diff = float(np.max(np.abs(gated_ea[enriched_items] - orig_ea[enriched_items])))

        result = {
            'n_enriched_products': len(enriched_items),
            'max_abs_diff_enriched_layer1': l1_diff,
            'max_abs_diff_enriched_full': full_diff,
            'tolerance': tol,
            'layer1_pass': l1_diff <= tol,
            'full_pass': full_diff <= tol,
        }
        if verbose:
            print('[Plan B enriched] enriched products: %d' % len(enriched_items))
            print('[Plan B enriched] LAYER 1   max|gated - original| = %.3e -> %s'
                  % (l1_diff, 'PASS' if result['layer1_pass'] else 'FAIL'))
            print('[Plan B enriched] FULL (%dL) max|gated - original| = %.3e -> %s'
                  % (self.n_layers, full_diff, 'PASS' if result['full_pass'] else 'FAIL'))
        return result

    def build_planCplus_diagnostic(self):
        # Independent pure-KGAT reference embedding (SAME weights, SAME A_in) used to
        # prove invariant (c): the KGAT propagation is byte-identical.
        _ua_ref, ea_ref = super(CRHKGE, self)._create_bi_interaction_embed()
        self.cr_dbg_ea_kgat_ref = ea_ref
        return self

    def check_planCplus_invariants(self, sess, tol=1e-6, verbose=True):
        """Plan C+ invariants (a)(b)(c) on a synthetic forward (dropout = 0).

        (a) standard products: e_final == e_kgat exactly.
        (c) KGAT propagation byte-identical to original KGAT (pre-residual stream
            equals an independent pure-KGAT stream with the same weights).
        (b) is the gamma=0 floor: run this with --cr_residual_gamma 0
            --cr_use_contrastive 0 and EVERY product (not just standard) matches
            e_kgat. Reported as the all-product diff here.

        Asserts (a) and (c); returns a dict. Does not require training.
        """
        if not hasattr(self, 'cr_dbg_ea_kgat_ref'):
            self.build_planCplus_diagnostic()

        feed = {
            self.mess_dropout: [0.] * self.n_layers,
            self.node_dropout: [0.] * self.n_layers,
        }
        ea_kgat, ea_final, ea_ref = sess.run(
            [self.cr_dbg_ea_kgat, self.ea_embeddings, self.cr_dbg_ea_kgat_ref],
            feed_dict=feed)

        enriched = set(int(i) for i in self.cr_config.get('enriched_product_ids', []))
        standard = [i for i in range(self.n_items) if i not in enriched]
        enriched_items = sorted(enriched)
        all_items = list(range(self.n_items))

        def _maxdiff(rows):
            if not rows:
                return 0.0
            return float(np.max(np.abs(ea_final[rows] - ea_kgat[rows])))

        a_diff = _maxdiff(standard)                         # invariant (a)
        b_diff = _maxdiff(all_items)                        # invariant (b) floor
        enr_diff = _maxdiff(enriched_items)                 # residual magnitude
        c_diff = float(np.max(np.abs(ea_kgat - ea_ref)))    # invariant (c)

        result = {
            'n_enriched_products': len(enriched_items),
            'n_standard_products': len(standard),
            'invariant_a_standard_eq_kgat': a_diff,
            'invariant_b_allproducts_eq_kgat_floor': b_diff,
            'invariant_c_propagation_eq_kgat': c_diff,
            'residual_magnitude_on_enriched': enr_diff,
            'tolerance': tol,
            'residual_gamma_init': self.cr_residual_gamma_init,
        }
        if verbose:
            print('[Plan C+ inv a] standard products == KGAT     : max diff %.3e -> %s'
                  % (a_diff, 'PASS' if a_diff <= tol else 'FAIL'))
            print('[Plan C+ inv c] KGAT propagation byte-identical: max diff %.3e -> %s'
                  % (c_diff, 'PASS' if c_diff <= tol else 'FAIL'))
            print('[Plan C+ inv b] ALL products == KGAT (floor; only when gamma=0): max diff %.3e %s'
                  % (b_diff, '-> PASS' if b_diff <= tol else '(expected > 0 when gamma != 0)'))
            print('[Plan C+      ] residual magnitude on enriched : %.3e (gamma_init=%.3f)'
                  % (enr_diff, self.cr_residual_gamma_init))

        assert a_diff <= tol, (
            'Invariant (a) VIOLATED: standard products deviate from KGAT by %.3e > %.1e'
            % (a_diff, tol))
        assert c_diff <= tol, (
            'Invariant (c) VIOLATED: KGAT propagation differs by %.3e > %.1e'
            % (c_diff, tol))
        return result

    def export_artifacts(self, sess, args, data_generator, final_perf):
        # Fase akhir setelah training:
        # export artifact untuk serving/retrieval dan integrasi dengan sistem
        # conversational. File yang dihasilkan berisi embedding produk,
        # embedding entitas, bobot relasi, path KG, dan konfigurasi encoder.
        export_feed = {
            self.mess_dropout: [0.] * self.n_layers,
            self.node_dropout: [0.] * self.n_layers,
        }
        user_embeddings, entity_embeddings = sess.run(
            [self.ua_embeddings, self.ea_embeddings],
            feed_dict=export_feed)

        relation_weight_rows = self._relation_weight_rows(sess)

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        artifact_dir = os.path.abspath(os.path.join(
            getattr(args, 'cr_artifact_path', '../artifacts/cr_hkge'),
            args.dataset,
            '%s_%s' % (self.model_type, timestamp)
        ))
        os.makedirs(artifact_dir, exist_ok=True)

        product_meta = self._load_product_metadata(getattr(data_generator, 'path', ''))
        entity_meta = self._load_entity_metadata(getattr(data_generator, 'path', ''))

        self._write_embeddings_tsv(
            os.path.join(artifact_dir, 'product_embeddings.tsv'),
            entity_embeddings[:self.n_items],
            product_meta,
            entity_meta,
            entity_kind='product')

        self._write_embeddings_tsv(
            os.path.join(artifact_dir, 'entity_embeddings.tsv'),
            entity_embeddings,
            product_meta,
            entity_meta,
            entity_kind='entity')

        self._write_relation_weights(
            os.path.join(artifact_dir, 'relation_weights.tsv'),
            relation_weight_rows)

        self._write_kg_paths(
            os.path.join(artifact_dir, 'kg_paths.jsonl'),
            data_generator,
            entity_meta)

        self._write_query_encoder_config(
            os.path.join(artifact_dir, 'query_encoder_config.json'),
            int(entity_embeddings.shape[1]))

        model_config = {
            'model_version': self.cr_model_version,
            'model_type': self.model_type,
            'dataset': args.dataset,
            'n_users': int(self.n_users),
            'n_items': int(self.n_items),
            'n_entities': int(self.n_entities),
            'n_relations_expanded': int(self.n_relations),
            'embedding_dim_final': int(entity_embeddings.shape[1]),
            'user_embedding_dim_final': int(user_embeddings.shape[1]),
            'cr_use_relation_weight': self.cr_use_relation_weight,
            'cr_use_cross_ref': self.cr_use_cross_ref,
            'cr_relation_weight_mode': self.cr_relation_weight_mode,
            'cr_relation_prior_mode': self.cr_relation_prior_mode,
            'cr_relation_prior_strength': self.cr_relation_prior_strength,
            'cr_relation_attention_scale': self.cr_relation_attention_scale,
            'cr_relation_aware_message': self.cr_relation_aware_message,
            'cr_relation_message_scale': self.cr_relation_message_scale,
            'cr_planB_gate': self.cr_planB_gate,
            'cr_use_residual': self.cr_use_residual,
            'cr_residual_gamma_init': self.cr_residual_gamma_init,
            'cr_residual_gamma_value': (float(sess.run(self.cr_residual_gamma))
                                        if self.cr_use_residual else 0.0),
            'cr_use_contrastive': self.cr_use_contrastive,
            'cr_contrastive_weight': self.cr_contrastive_weight,
            'cr_contrastive_margin': self.cr_contrastive_margin,
            'cr_contrastive_negs': self.cr_contrastive_negs,
            'cr_n_contrastive_pairs': int(getattr(self, 'cr_n_contrastive_pairs', 0)),
            'cr_cross_ref_alpha': self.cr_cross_ref_alpha,
            'cr_cross_ref_bi_interaction': self.cr_cross_ref_bi_interaction,
            'cr_cross_ref_gate': self.cr_cross_ref_gate_enabled,
            'cr_cross_ref_gate_init': self.cr_cross_ref_gate_init,
            'cr_cross_ref_gate_value': self._cross_ref_gate_value(sess),
            'cr_best_metric': getattr(args, 'cr_best_metric', 'ndcg'),
            'cr_best_k': int(getattr(args, 'cr_best_k', 3)),
            'cr_export_best_checkpoint': bool(int(getattr(args, 'cr_export_best_checkpoint', 1))),
            'cr_cross_ref_attention': 'strict_neighbor_attention',
            'cr_global_attr_attention_edges': int(getattr(self, 'cr_global_attr_attention_edge_count', 0)),
            'enriched_product_count': int(len(self.cr_config.get('enriched_product_ids', []))),
            'global_attr_relation_ids': [int(r) for r in self.cr_global_attr_relation_ids],
            'query_encoder_config': 'query_encoder_config.json',
            'final_performance': final_perf,
        }

        with open(os.path.join(artifact_dir, 'model_config.json'), 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2)

        print('export CR-HKGE artifacts in path: ', artifact_dir)
        return artifact_dir

    def _cross_ref_gate_value(self, sess):
        if not self.cr_use_cross_ref:
            return 0.0
        return float(sess.run(self.cr_cross_ref_gate))

    def _relation_weight_rows(self, sess):
        if self.cr_use_relation_weight:
            probs, multipliers, message_multipliers = sess.run([
                self.cr_relation_type_probs,
                self.cr_relation_type_multipliers,
                self.cr_relation_type_message_multipliers
            ])
        else:
            probs = np.ones(self.cr_n_relation_types, dtype=np.float32) / float(self.cr_n_relation_types)
            multipliers = np.ones(self.cr_n_relation_types, dtype=np.float32)
            message_multipliers = np.ones(self.cr_n_relation_types, dtype=np.float32)

        rows = []
        for type_id, type_name in enumerate(self.cr_relation_type_names):
            rows.append({
                'relation_type_id': type_id,
                'relation_type_name': type_name,
                'probability': float(probs[type_id]),
                'multiplier': float(multipliers[type_id]),
                'message_multiplier': float(message_multipliers[type_id]),
            })
        return rows

    def _load_product_metadata(self, dataset_path):
        product_file = os.path.join(dataset_path, 'product2id.tsv')
        product_meta = {}
        if not os.path.exists(product_file):
            return product_meta

        with open(product_file, 'r', encoding='utf-8') as f:
            next(f, None)
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 3:
                    product_meta[int(parts[0])] = {
                        'old_entity_id': parts[1],
                        'name': parts[2],
                    }
        return product_meta

    def _load_entity_metadata(self, dataset_path):
        entity_file = os.path.join(dataset_path, 'entity2id_typed.tsv')
        entity_meta = {}
        if not os.path.exists(entity_file):
            return entity_meta

        with open(entity_file, 'r', encoding='utf-8') as f:
            next(f, None)
            for line in f:
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 4:
                    entity_meta[int(parts[0])] = {
                        'old_entity_id': parts[1],
                        'type': parts[2],
                        'name': parts[3],
                    }
        return entity_meta

    def _embedding_to_json(self, embedding):
        return json.dumps(
            [round(float(value), 8) for value in embedding.tolist()],
            separators=(',', ':'))

    def _write_embeddings_tsv(self, path, embeddings, product_meta, entity_meta, entity_kind):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('entity_id\told_entity_id\tentity_type\tentity_name\tembedding_dim\tembedding_json\tmodel_version\n')
            for entity_id, embedding in enumerate(embeddings):
                meta = entity_meta.get(entity_id, {})
                if entity_kind == 'product':
                    product = product_meta.get(entity_id, {})
                    entity_name = product.get('name', meta.get('name', 'entity_%d' % entity_id))
                    old_entity_id = product.get('old_entity_id', meta.get('old_entity_id', ''))
                    entity_type = 'product'
                else:
                    entity_name = meta.get('name', 'entity_%d' % entity_id)
                    old_entity_id = meta.get('old_entity_id', '')
                    entity_type = meta.get('type', '')

                f.write('%d\t%s\t%s\t%s\t%d\t%s\t%s\n' % (
                    entity_id,
                    old_entity_id,
                    entity_type,
                    entity_name,
                    len(embedding),
                    self._embedding_to_json(embedding),
                    self.cr_model_version))

    def _write_relation_weights(self, path, rows):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('relation_type_id\trelation_type_name\tprobability\tmultiplier\tmessage_multiplier\tmodel_version\n')
            for row in rows:
                f.write('%d\t%s\t%.8f\t%.8f\t%.8f\t%s\n' % (
                    row['relation_type_id'],
                    row['relation_type_name'],
                    row['probability'],
                    row['multiplier'],
                    row['message_multiplier'],
                    self.cr_model_version))

    def _write_query_encoder_config(self, path, embedding_dim):
        config = {
            'model_version': self.cr_model_version,
            'embedding_dim': int(embedding_dim),
            'entity_matching': {
                'accords': ['accord', 'global_accord'],
                'family': ['family', 'global_family'],
                'notes': ['note'],
                'visual_notes': ['note'],
                'reference': ['global_ref'],
                'inspired_by': ['global_ref'],
            },
            'field_relation_map': {
                'accords': {
                    'accord': 'has_accord',
                    'global_accord': 'has_global_accord',
                },
                'family': {
                    'family': 'belongs_to_family',
                    'global_family': 'belongs_to_global_family',
                },
                'notes': {
                    'note': 'has_visual_note',
                },
                'visual_notes': {
                    'note': 'has_visual_note',
                },
                'reference': {
                    'global_ref': 'inspired_by',
                },
                'inspired_by': {
                    'global_ref': 'inspired_by',
                },
            },
            'kg_path_matching': {
                'policy': 'relation_compatible',
                'allow_name_match': True,
                'relation_compatible_entity_types': {
                    'has_accord': ['accord'],
                    'has_global_accord': ['global_accord'],
                    'belongs_to_family': ['family'],
                    'belongs_to_global_family': ['global_family'],
                    'has_visual_note': ['note'],
                    'inspired_by': ['global_ref'],
                },
            },
            'retrieval_rerank': {
                'query_aware_rerank': True,
                'candidate_pool': 50,
                'min_matched_paths': 1,
                'match_bonus': 0.05,
            },
            'relation_weights_used': bool(self.cr_use_relation_weight),
            'relation_weight_mode': self.cr_relation_weight_mode,
            'relation_weight_file': 'relation_weights.tsv',
            'product_embedding_file': 'product_embeddings.tsv',
            'entity_embedding_file': 'entity_embeddings.tsv',
            'kg_path_file': 'kg_paths.jsonl',
            'aggregation': 'weighted_mean',
            'normalization': 'l2',
            'score_function': 'cosine',
            'top_k': 3,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    def _write_kg_paths(self, path, data_generator, entity_meta):
        relation_id_to_name = self.cr_config.get('relation_id_to_name', {})
        product_paths = {}

        for head, relation, tail in data_generator.kg_data:
            head = int(head)
            relation = int(relation)
            tail = int(tail)
            if head >= self.n_items:
                continue

            head_meta = entity_meta.get(head, {})
            tail_meta = entity_meta.get(tail, {})
            product_paths.setdefault(head, []).append({
                'head_entity_id': head,
                'head_entity_type': head_meta.get('type', 'product'),
                'head_entity_name': head_meta.get('name', 'entity_%d' % head),
                'relation_id': relation,
                'relation_name': relation_id_to_name.get(relation, 'relation_%d' % relation),
                'relation_scope': 'product',
                'tail_entity_id': tail,
                'tail_entity_type': tail_meta.get('type', ''),
                'tail_entity_name': tail_meta.get('name', 'entity_%d' % tail),
            })

        product_to_global_ref = self.cr_config.get('product_to_global_ref', {})
        global_ref_to_attributes = self.cr_config.get('global_ref_to_attributes', {})

        for product_id, global_refs in product_to_global_ref.items():
            product_id = int(product_id)
            for global_ref_id in global_refs:
                global_ref_id = int(global_ref_id)
                attrs = global_ref_to_attributes.get(global_ref_id)
                if attrs is None:
                    attrs = global_ref_to_attributes.get(str(global_ref_id), [])

                head_meta = entity_meta.get(global_ref_id, {})
                for relation, tail in attrs:
                    relation = int(relation)
                    tail = int(tail)
                    tail_meta = entity_meta.get(tail, {})
                    product_paths.setdefault(product_id, []).append({
                        'head_entity_id': global_ref_id,
                        'head_entity_type': head_meta.get('type', 'global_ref'),
                        'head_entity_name': head_meta.get('name', 'entity_%d' % global_ref_id),
                        'relation_id': relation,
                        'relation_name': relation_id_to_name.get(relation, 'relation_%d' % relation),
                        'relation_scope': 'global_reference',
                        'tail_entity_id': tail,
                        'tail_entity_type': tail_meta.get('type', ''),
                        'tail_entity_name': tail_meta.get('name', 'entity_%d' % tail),
                    })

        with open(path, 'w', encoding='utf-8') as f:
            for product_id in sorted(product_paths.keys()):
                entity_meta_row = entity_meta.get(product_id, {})
                row = {
                    'product_id': product_id,
                    'product_name': entity_meta_row.get('name', 'product_%d' % product_id),
                    'model_version': self.cr_model_version,
                    'kg_path': product_paths[product_id],
                }
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
