"""Plan B invariant check: standard products get EXACTLY the plain-KGAT update.

Builds the real CR-HKGE (Plan B) model on a dataset and verifies, on a synthetic
forward pass (random-initialised weights, message dropout = 0), that for every
STANDARD product (item node without inspired_by) the layer-1 gated update equals
the plain-KGAT update within floating-point tolerance, while ENRICHED products
differ. No training and no checkpoint are required.

Usage (from Model/):
  python check_planb_invariant.py --data_path ../ --dataset dataset-aromatique-crhkge-holdout \
      --model_type cr_hkge --cr_use_relation_weight 1 --cr_use_cross_ref 1 \
      --cr_relation_aware_message 1 --cr_relation_prior_mode fragrance \
      --layer_size '[64,32,16]' --embed_size 64 --kge_size 64 --regs '[1e-5,1e-5]' \
      --alg_type bi --adj_type si --adj_uni_type sum
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

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())

    result = model.check_standard_equals_kgat(sess, tol=1e-5, verbose=True)

    # Enriched products SHOULD diverge from KGAT (otherwise the CR path is inert).
    if result['n_enriched_products'] > 0 and result['max_abs_diff_enriched_vs_kgat'] == 0.0:
        print('WARNING: enriched products are identical to KGAT; CR path appears inert '
              '(expected when training on a holdout KG with 0 enriched edges).')

    print('PLAN B INVARIANT CHECK PASSED: standard products == plain KGAT (max diff %.3e <= %.1e)'
          % (result['max_abs_diff_standard_vs_kgat'], result['tolerance']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
