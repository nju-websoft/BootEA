import multiprocessing

import gc
import numpy as np
import time

from param import P
from utils import div_list

g = 1000000000


def eval_alignment(sim_mat, top_k, mess=""):
    t = time.time()
    ref_num = sim_mat.shape[0]
    num = [0 for k in top_k]
    mean = 0
    mrr = 0
    for i in range(ref_num):
        rank = (-sim_mat[i, :]).argsort()
        assert i in rank
        rank_index = np.where(rank == i)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
    acc = np.array(num) / ref_num
    for i in range(len(acc)):
        acc[i] = round(acc[i], 4)
    mean /= ref_num
    mrr /= ref_num
    if mess == "":
        print(
            "hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(top_k, acc, mean, mrr, time.time() - t))
    else:
        print("{}, hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(mess, top_k, acc, mean, mrr,
                                                                                     time.time() - t))
    return mean / ref_num, acc[2]


def cal_rank(task, sim, top_k):
    mean = 0
    mrr = 0
    num = [0 for k in top_k]
    for i in range(len(task)):
        ref = task[i]
        rank = (-sim[i, :]).argsort()
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
        # del rank
    return mean, mrr, num


def eval_alignment_mul(sim_mat, top_k, mess=""):
    t = time.time()
    ref_num = sim_mat.shape[0]
    t_num = [0 for k in top_k]
    t_mean = 0
    t_mrr = 0
    tasks = div_list(np.array(range(ref_num)), P.nums_threads)
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_rank, (task, sim_mat[task, :], top_k)))
    pool.close()
    pool.join()

    for res in reses:
        mean, mrr, num = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += np.array(num)

    acc = np.array(t_num) / ref_num
    for i in range(len(acc)):
        acc[i] = round(acc[i], 4)
    t_mean /= ref_num
    t_mrr /= ref_num
    print("{}, hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(mess, top_k, acc, t_mean, t_mrr,
                                                                                 time.time() - t))


def cal_rank_multi_embed(frags, dic, sub_embed, embed, top_k):
    mean = 0
    mrr = 0
    num = np.array([0 for k in top_k])
    mean1 = 0
    mrr1 = 0
    num1 = np.array([0 for k in top_k])
    sim_mat = np.matmul(sub_embed, embed.T)
    prec_set = set()
    aligned_e = None
    for i in range(len(frags)):
        ref = frags[i]

        rank = (-sim_mat[i, :]).argsort()
        aligned_e = rank[0]
        assert ref in rank
        rank_index = np.where(rank == ref)[0][0]
        mean += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                num[j] += 1
        # del rank

        if dic is not None and dic.get(ref, -1) > -1:
            e2 = dic.get(ref)
            sim_mat[i, e2] += 1.0
            rank = (-sim_mat[i, :]).argsort()
            aligned_e = rank[0]
            assert ref in rank
            rank_index = np.where(rank == ref)[0][0]
            mean1 += (rank_index + 1)
            mrr1 += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    num1[j] += 1
            # del rank
        else:
            mean1 += (rank_index + 1)
            mrr1 += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    num1[j] += 1

        prec_set.add((ref, aligned_e))

    del sim_mat
    gc.collect()
    return mean, mrr, num, mean1, mrr1, num1, prec_set


def eval_alignment_multi_embed(embed1, embed2, top_k, selected_pairs, mess=""):
    def pair2dic(pairs):
        if pairs is None or len(pairs) == 0:
            return None
        dic = dict()
        for i, j in pairs:
            if i not in dic.keys():
                dic[i] = j
        assert len(dic) == len(pairs)
        return dic

    t = time.time()
    dic = pair2dic(selected_pairs)
    ref_num = embed1.shape[0]
    t_num = np.array([0 for k in top_k])
    t_mean = 0
    t_mrr = 0
    t_num1 = np.array([0 for k in top_k])
    t_mean1 = 0
    t_mrr1 = 0
    t_prec_set = set()
    frags = div_list(np.array(range(ref_num)), P.nums_threads)
    pool = multiprocessing.Pool(processes=len(frags))
    reses = list()
    for frag in frags:
        reses.append(pool.apply_async(cal_rank_multi_embed, (frag, dic, embed1[frag, :], embed2, top_k)))
    pool.close()
    pool.join()

    for res in reses:
        mean, mrr, num, mean1, mrr1, num1, prec_set = res.get()
        t_mean += mean
        t_mrr += mrr
        t_num += num
        t_mean1 += mean1
        t_mrr1 += mrr1
        t_num1 += num1
        t_prec_set |= prec_set

    assert len(t_prec_set) == ref_num

    acc = t_num / ref_num
    for i in range(len(acc)):
        acc[i] = round(acc[i], 4)
    t_mean /= ref_num
    t_mrr /= ref_num
    print("{}, hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(mess, top_k, acc, t_mean, t_mrr,
                                                                                 time.time() - t))
    if selected_pairs is not None and len(selected_pairs) > 0:
        acc1 = t_num1 / ref_num
        for i in range(len(acc1)):
            acc1[i] = round(acc1[i], 4)
        t_mean1 /= ref_num
        t_mrr1 /= ref_num
        print("{}, hits@{} = {}, mr = {:.3f}, mrr = {:.3f}, time = {:.3f} s ".format(mess, top_k, acc1, t_mean1, t_mrr1,
                                                                                     time.time() - t))
    return t_prec_set
