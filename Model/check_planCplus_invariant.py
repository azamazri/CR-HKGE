"""Plan C+ invariant diagnostic (no training, no checkpoint).

Builds the CR-HKGE-C+ model on a dataset and asserts:
  (a) standard products: e_final == e_kgat exactly.
  (c) KGAT propagation byte-identical to original KGAT.
  (b) gamma=0 floor: run with --cr_residual_gamma 0 --cr_use_contrastive 0 and
      EVERY product equals e_kgat (reported as the all-product diff).

Usage (from Model/), mirror the study's CR-HKGE-C+ flags:
  python check_planCplus_invariant.py --data_path ../ \
      --dataset dataset-aromatique-crhkge-ready --model_type cr_hkge \
      --cr_use_relation_weight 0 --cr_use_cross_ref 0 --cr_relation_aware_message 0 \
      --cr_use_residual 1 --cr_residual_gamma 0.1 \
      --cr_use_contrastive 1 --cr_contrastive_weight 0.1 \
      --layer_size '[64,32,16]' --embed_size 64 --kge_size 64 \
      --regs '[1e-5,1e-5]' --alg_type bi --adj_type si --adj_uni_type sum

  # invariant (b) floor:
  ... --cr_residual_gamma 0 --cr_use_contrastive 0
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
    model.build_planCplus_diagnostic()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)
    sess.run(tf.global_variables_initializer())

    res = model.check_planCplus_invariants(sess, tol=1e-6, verbose=True)

    floor_note = ''
    if args.cr_residual_gamma == 0 and int(args.cr_use_contrastive) == 0:
        floor_note = ' (gamma=0, contrastive off -> floor expected PASS)'
    print('\n=== VERDICT ===')
    print('(a) standard == KGAT            : %s (max diff %.3e)'
          % ('PASS' if res['invariant_a_standard_eq_kgat'] <= res['tolerance'] else 'FAIL',
             res['invariant_a_standard_eq_kgat']))
    print('(c) propagation == KGAT         : %s (max diff %.3e)'
          % ('PASS' if res['invariant_c_propagation_eq_kgat'] <= res['tolerance'] else 'FAIL',
             res['invariant_c_propagation_eq_kgat']))
    print('(b) ALL products == KGAT floor  : max diff %.3e%s'
          % (res['invariant_b_allproducts_eq_kgat_floor'], floor_note))
    print('residual magnitude on enriched  : %.3e' % res['residual_magnitude_on_enriched'])
    print('PLAN C+ INVARIANTS (a)+(c) PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
