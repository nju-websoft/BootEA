import random
import time
import math

import gc

import psutil
import tensorflow as tf
import numpy as np

from sklearn import preprocessing
from param import *
from test_funcs import eval_alignment, eval_alignment_mul, eval_alignment_multi_embed

g = 1000000000


def embed_init(mat_x, mat_y, name, is_l2=False):
    print("embed_init")
    embeddings = tf.Variable(tf.truncated_normal([mat_x, mat_y], stddev=1.0 / math.sqrt(P.embed_size)))
    return tf.nn.l2_normalize(embeddings, 1) if is_l2 else embeddings


def mul(tensor1, tensor2, session, num, sigmoid):
    t = time.time()
    if num < 20000:
        sim_mat = tf.matmul(tensor1, tensor2, transpose_b=True)
        if sigmoid:
            res = tf.sigmoid(sim_mat).eval(session=session)
        else:
            res = sim_mat.eval(session=session)
    else:
        res = np.matmul(tensor1.eval(session=session), tensor2.eval(session=session).T)
    print("mat mul costs: {:.3f}".format(time.time() - t))
    return res


class KGE_Model:
    def __init__(self, ent_num, rel_num, seed_sup_ent1, seed_sup_ent2, ref_ent1, ref_ent2, kb1_ents, kb2_ents, dim, embed_size, lr):
        self.ent_num = ent_num
        self.rel_num = rel_num

        self.seed_sup_ent1 = seed_sup_ent1
        self.seed_sup_ent2 = seed_sup_ent2
        self.ref_ent1 = ref_ent1
        self.ref_ent2 = ref_ent2
        self.kb1_ents = kb1_ents
        self.kb2_ents = kb2_ents

        self.embed_size = embed_size
        self.lr = lr
        self.dim = dim

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        self._generate_variables()
        self._generate_graph()
        self._generate_alignment_graph()
        self._generate_likelihood_graph()

        tf.global_variables_initializer().run(session=self.session)

    def _generate_variables(self):
        with tf.variable_scope('relation' + 'embedding'):
            self.ent_embeddings = embed_init(self.ent_num, self.embed_size, "ent_embeds")
            self.rel_embeddings = embed_init(self.rel_num, self.embed_size, "rel_embeds")
            self.ent_embeddings = tf.nn.l2_normalize(self.ent_embeddings, 1)
            self.rel_embeddings = tf.nn.l2_normalize(self.rel_embeddings, 1)

    def _generate_graph(self):
        def generate_loss(phs, prs, pts, nhs, nrs, nts):
            pos_score = tf.reduce_sum(tf.pow(phs + prs - pts, 2), 1)
            neg_score = tf.reduce_sum(tf.pow(nhs + nrs - nts, 2), 1)
            pos_loss = tf.reduce_sum(tf.maximum(pos_score - tf.constant(P.lambda_1), 0))
            neg_loss = P.mu_1 * tf.reduce_sum(tf.maximum(tf.constant(P.lambda_2) - neg_score, 0))

            return pos_loss, neg_loss

        def generate_optimizer(loss):
            opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("relation")]
            optimizer = tf.train.AdagradOptimizer(self.lr).minimize(loss, var_list=opt_vars)
            return optimizer

        self.pos_hs = tf.placeholder(tf.int32, shape=[None])
        self.pos_rs = tf.placeholder(tf.int32, shape=[None])
        self.pos_ts = tf.placeholder(tf.int32, shape=[None])
        self.neg_hs = tf.placeholder(tf.int32, shape=[None])
        self.neg_rs = tf.placeholder(tf.int32, shape=[None])
        self.neg_ts = tf.placeholder(tf.int32, shape=[None])
        phs = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_hs)
        prs = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_rs)
        pts = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_ts)
        nhs = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_hs)
        nrs = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_rs)
        nts = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_ts)
        self.pos_loss, self.neg_loss = generate_loss(phs, prs, pts, nhs, nrs, nts)
        self.triple_loss = self.pos_loss + self.neg_loss
        self.triple_optimizer = generate_optimizer(self.triple_loss)

    def _generate_alignment_graph(self):
        def generate_loss(phs, prs, pts):
            pos_loss = tf.reduce_sum(tf.log(tf.sigmoid(-tf.reduce_sum(tf.pow(phs + prs - pts, 2), 1))))
            return - pos_loss

        def generate_optimizer(loss):
            opt_vars = [v for v in tf.trainable_variables() if v.name.startswith("relation")]
            optimizer = tf.train.AdagradOptimizer(self.lr).minimize(loss, var_list=opt_vars)
            return optimizer

        self.new_h = tf.placeholder(tf.int32, shape=[None])
        self.new_r = tf.placeholder(tf.int32, shape=[None])
        self.new_t = tf.placeholder(tf.int32, shape=[None])
        phs = tf.nn.embedding_lookup(self.ent_embeddings, self.new_h)
        prs = tf.nn.embedding_lookup(self.rel_embeddings, self.new_r)
        pts = tf.nn.embedding_lookup(self.ent_embeddings, self.new_t)
        self.alignment_loss = generate_loss(phs, prs, pts)
        self.alignment_optimizer = generate_optimizer(self.alignment_loss)

    def _generate_likelihood_graph(self):
        self.ents1 = tf.placeholder(tf.int32, shape=[None])
        self.ents2 = tf.placeholder(tf.int32, shape=[None])
        self.likelihood_mat = tf.placeholder(tf.float32, shape=[self.dim, self.dim])

        ent1_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.ents1)
        ent2_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.ents2)
        mat = tf.log(tf.sigmoid(tf.matmul(ent1_embed, ent2_embed, transpose_b=True)))
        self.likelihood_loss = -tf.reduce_sum(tf.multiply(mat, self.likelihood_mat))
        self.likelihood_optimizer = tf.train.AdagradOptimizer(self.lr).minimize(self.likelihood_loss)

    def test(self, selected_pairs=None):
        t1 = time.time()
        refs1_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.ref_ent1)
        refs2_embed = tf.nn.embedding_lookup(self.ent_embeddings, self.ref_ent2)
        refs1_embed = tf.nn.l2_normalize(refs1_embed, 1)
        refs2_embed = tf.nn.l2_normalize(refs2_embed, 1)
        refs1_embed = refs1_embed.eval(session=self.session)
        refs2_embed = refs2_embed.eval(session=self.session)
        prec_set = eval_alignment_multi_embed(refs1_embed, refs2_embed, P.ent_top_k, selected_pairs, mess="ent alignment")
        t2 = time.time()
        m1 = psutil.virtual_memory().used
        del refs1_embed, refs2_embed
        gc.collect()
        # print("gc costs {:.3f} s, mem change {:.6f} G".format(time.time() - t2,
        #                                                       (psutil.virtual_memory().used - m1) / g))
        print("testing ent alignment costs: {:.3f} s\n".format(time.time() - t1))
        return prec_set

    def eval_ent_embeddings(self):
        return self.ent_embeddings.eval(session=self.session)

    def eval_rel_embeddings(self):
        return self.rel_embeddings.eval(session=self.session)

    def eval_ref_sim_mat(self, sigmoid=False):
        refs1_embeddings = tf.nn.embedding_lookup(self.ent_embeddings, self.ref_ent1)
        refs2_embeddings = tf.nn.embedding_lookup(self.ent_embeddings, self.ref_ent2)
        refs1_embeddings = tf.nn.l2_normalize(refs1_embeddings, 1)
        refs2_embeddings = tf.nn.l2_normalize(refs2_embeddings, 1)
        return mul(refs1_embeddings, refs2_embeddings, self.session, len(self.ref_ent1), sigmoid)

    def eval_kb1_mat(self, sigmoid=False):
        ent1_embeddings = tf.nn.embedding_lookup(self.ent_embeddings, self.kb1_ents)
        ent2_embeddings = tf.nn.embedding_lookup(self.ent_embeddings, self.kb1_ents)
        return mul(ent1_embeddings, ent2_embeddings, self.session, len(self.kb1_ents), sigmoid)

    def eval_kb2_mat(self, sigmoid=False):
        ent1_embeddings = tf.nn.embedding_lookup(self.ent_embeddings, self.kb2_ents)
        ent2_embeddings = tf.nn.embedding_lookup(self.ent_embeddings, self.kb2_ents)
        return mul(ent1_embeddings, ent2_embeddings, self.session, len(self.kb2_ents), sigmoid)

    def eval_sim_mat(self, ent1, ent2, sigmoid=False):
        embeddings1 = tf.nn.embedding_lookup(self.ent_embeddings, ent1)
        embeddings2 = tf.nn.embedding_lookup(self.ent_embeddings, ent2)
        return mul(embeddings1, embeddings2, self.session, len(ent1), sigmoid)

    def eval_kb1_embed(self):
        return tf.nn.embedding_lookup(self.ent_embeddings, self.kb1_ents).eval(session=self.session)

    def eval_kb2_embed(self):
        return tf.nn.embedding_lookup(self.ent_embeddings, self.kb2_ents).eval(session=self.session)

    def save(self, folder, suffix):
        np.save(folder + 'ent1_embeds_' + suffix + '.npy', self.eval_kb1_embed())
        np.save(folder + 'ent2_embeds_' + suffix + '.npy', self.eval_kb2_embed())
