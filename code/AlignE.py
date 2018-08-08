import sys
import time
import utils as ut

from train_funcs import get_model, train_tris_k_epo, train_tris_1epo
from model import P


def train(folder):
    ori_triples1, ori_triples2, triples1, triples2, model = get_model(folder)
    hits1 = None
    if P.epsilon > 0:  # using epsilon-truncated uniform negative sampling
        trunc_ent_num = int(len(ori_triples1.ent_list) * (1 - P.epsilon))
        assert trunc_ent_num > 0
        print("trunc ent num:", trunc_ent_num)
        iters = 5 if "15" in folder else 10
        for t in range(1, P.epochs // iters + 1):
            print("iteration ", t)
            hits1 = train_tris_k_epo(model, triples1, triples2, iters, trunc_ent_num, None, None)

    else:  # using epsilon-truncated uniform negative sampling
        test = 10 if "15" in folder else 50
        for epo in range(1, P.epochs + 1):
            loss, t = train_tris_1epo(model, triples1, triples2, None, None)
            print("epoch {}: triple_loss = {:.3f}, time = {:.3f} s".format(epo, loss, t))
            if epo % test == 0:
                hits1 = model.test()

    # save embeddings and hits@1 results
    ut.pair2file(folder + "hits1_results_AlignE_trunc" + str(P.epsilon), hits1)
    model.save(folder, "AlignE_trunc" + str(P.epsilon))


if __name__ == '__main__':
    t = time.time()
    if len(sys.argv) == 2:
        folder = sys.argv[1]
    else:
        # folder = '../DBP15K/zh_en/mtranse/0_3/'
        folder = '../dataset/DWY100K/dbp_wd/mapping/0_3/'
    train(folder)
    print("total time = {:.3f} s".format(time.time() - t))
