"""Plan B ENRICHED invariant diagnostic (no training, no checkpoint).

Builds the gated CR-HKGE-B model AND a parallel ungated stream (== original
CR-HKGE) sharing the SAME random weights, then reports, for ENRICHED products:
  - layer-1 gated vs original  (expected PASS: gate does not touch enriched)
  - full multi-layer gated vs original (the tensor that feeds predictions)
It also re-checks standard == plain KGAT.

Usage (from Model/), mirror the study's CR-HKGE-PlanB flags:
  python check_planb_enriched_invariant.py --data_path ../ \
      --dataset dataset-aromatique-crhkge-ready \
      --model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 \
      --cr_relation_aware_message 1 --cr_relation_prior_mode fragrance \
      --cr_cross_ref_alpha 0.1 --cr_planB_gate 1 \
      --layer_size '[64,32,16]' --embed_size 64 --kge_size 64 \
      --regs '[1e-5,1e-5]' --alg_type bi --adj_type si --adj_uni_type sum
"""

import sys

from utility.tf_compat import tf
from utility.parser import parse_args
from utility.loader_kgat import KGAT_loader
from CRHKGE import CRHKGE


def main():
    args = parse_args()
    tf.set_random_seed(args.seed)

    path = args.data_path + args.dataset
    data_generator = KGAT_loader(args=args, path=path)

    config = {
        'n_users': data_generator.n_users,
        'n_items': data_generator.n_items,
        'n_entities': data_generator.n_entities,
        'n_relations': data_generator.n_relations,
        'A_in': sum(data_generator.lap_list),
        'all_h_list': data_generator.all_h_list,
        'all_r_list': data_generator.all_r_list,
        'all_t_list': data_generator.all_t_list,
        'all_v_list': data_generator.all_v_list,
        'cr_hkge_config': data_generator.get_cr_hkge_config(),
        'lap_list': data_generator.lap_list,
        'adj_r_list': data_generator.adj_r_list,
    }

    model = CRHKGE(data_config=config, pretrain_data=None, args=args)
    model.build_planB_enriched_diagnostic()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())

    print('=== Standard products vs plain KGAT ===')
    std = model.check_standard_equals_kgat(sess, tol=1e-5, verbose=True)

    print('\n=== Enriched products: gated (Plan B) vs original CR-HKGE ===')
    enr = model.check_enriched_equals_original(sess, tol=1e-5, verbose=True)

    print('\n=== VERDICT ===')
    print('standard == KGAT (layer 1)          : %s (max diff %.3e)'
          % ('PASS' if std['max_abs_diff_standard_vs_kgat'] <= std['tolerance'] else 'FAIL',
             std['max_abs_diff_standard_vs_kgat']))
    print('enriched == original (layer 1)      : %s (max diff %.3e)'
          % ('PASS' if enr['layer1_pass'] else 'FAIL', enr['max_abs_diff_enriched_layer1']))
    print('enriched == original (full %d layers): %s (max diff %.3e)'
          % (model.n_layers, 'PASS' if enr['full_pass'] else 'FAIL',
             enr['max_abs_diff_enriched_full']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
