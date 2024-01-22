import copy
import itertools
import numpy as np
from tqdm import tqdm
from os.path import join
from datetime import datetime

from eval_metrics import eval_sysu, eval_regdb, eval_llcm


def print_time(text):
    time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('[{}] {}'.format(time, text))


class Distmat:
    def __init__(self, dataset, feat_dir, method):
        assert dataset in ('sysu', 'regdb', 'llcm')
        self.dataset = dataset
        self.feat_dir = feat_dir
        self.method = method
        self.alpha, self.beta = 1/2, 2/3
        print_time('Start ReRanking on Dataset [{}] with Method [{}]'.format(dataset.upper(), method.upper()))
        self.load_features()  # load all labels and features
        print_time('Features Loaded')
        self.compute_original_distmat()  # compute original distance
        print_time('OriDist Computed')

    def _load_npy(self, name, trial=None):
        if trial is not None:
            name += '_{}'.format(trial)
        name += '.npy'
        try:
            return np.load(join(self.feat_dir, name))
        except FileNotFoundError:
            return None

    def _compute_ori_dist(self, x, y, distance='cosine', norm=False):
        assert distance in ('cosine', 'euclidean')
        fn_norm = lambda x: x / np.sqrt(np.sum(x ** 2, axis=1))[:, np.newaxis]
        if norm:
            x, y = fn_norm(x), fn_norm(y)
        if distance == 'cosine':
            return 1 - np.dot(x, y.T)
        elif distance == 'euclidean':
            return np.sqrt(
                np.sum(x ** 2, axis=1)[:, np.newaxis] +
                np.sum(y ** 2, axis=1)[np.newaxis, :] -
                2 * np.dot(x, y.T) + 1e-5
            )

    def load_features(self):
        self.q_num = []
        self.g_num = []
        self.all_num = []
        self.query_label = []
        self.query_cam = []
        self.gallery_label = []
        self.gallery_cam = []
        self.all_feat_1 = []
        self.all_feat_2 = []
        self.all_feat_3 = []
        self.all_feat_4 = []
        self.all_feat_5 = []
        self.all_feat_6 = []
        self.all_feat_txt = []
        self.all_feat_joint = []

        if self.dataset in ('sysu', 'llcm'):
            query_feat_1 = self._load_npy('query_feat1')
            query_feat_2 = self._load_npy('query_feat2')
            query_feat_3 = self._load_npy('query_feat3')
            query_feat_4 = self._load_npy('query_feat4')
            query_feat_5 = self._load_npy('query_feat5')
            query_feat_6 = self._load_npy('query_feat6')
            query_feat_txt = self._load_npy('query_feat-txt')
            query_feat_joint = self._load_npy('query_feat-joint')
            self.query_label.append(self._load_npy('query_pid'))
            self.query_cam.append(self._load_npy('query_cam'))
            self.q_num = [query_feat_1.shape[0]] * 10

        for trial in range(10):
            # load features
            if self.dataset == 'regdb':
                query_feat_1 = self._load_npy('query_feat1', trial)
                query_feat_2 = self._load_npy('query_feat2', trial)
                query_feat_3 = self._load_npy('query_feat3', trial)
                query_feat_4 = self._load_npy('query_feat4', trial)
                query_feat_5 = self._load_npy('query_feat5', trial)
                query_feat_6 = self._load_npy('query_feat6', trial)
                query_feat_txt = self._load_npy('query_feat-txt', trial)
                query_feat_joint = self._load_npy('query_feat-joint', trial)
                self.query_label.append(self._load_npy('query_pid', trial))
                self.q_num.append(query_feat_1.shape[0])
            else:
                self.gallery_cam.append(self._load_npy('gallery_cam', trial))
            gallery_feat_1 = self._load_npy('gallery_feat1', trial)
            gallery_feat_2 = self._load_npy('gallery_feat2', trial)
            gallery_feat_3 = self._load_npy('gallery_feat3', trial)
            gallery_feat_4 = self._load_npy('gallery_feat4', trial)
            gallery_feat_5 = self._load_npy('gallery_feat5', trial)
            gallery_feat_6 = self._load_npy('gallery_feat6', trial)
            gallery_feat_txt = self._load_npy('gallery_feat-txt', trial)
            gallery_feat_joint = self._load_npy('gallery_feat-joint', trial)
            self.gallery_label.append(self._load_npy('gallery_pid', trial))
            self.g_num.append(gallery_feat_1.shape[0])
            # concatenation
            self.all_feat_1.append(np.concatenate([query_feat_1, gallery_feat_1], axis=0))
            self.all_feat_2.append(np.concatenate([query_feat_2, gallery_feat_2], axis=0))
            self.all_feat_3.append(np.concatenate([query_feat_3, gallery_feat_3], axis=0))
            self.all_feat_4.append(np.concatenate([query_feat_4, gallery_feat_4], axis=0))
            self.all_feat_5.append(np.concatenate([query_feat_5, gallery_feat_5], axis=0))
            self.all_feat_6.append(np.concatenate([query_feat_6, gallery_feat_6], axis=0))
            self.all_feat_txt.append(np.concatenate([query_feat_txt, gallery_feat_txt], axis=0))
            self.all_feat_joint.append(np.concatenate([query_feat_joint, gallery_feat_joint], axis=0))
            self.all_num.append(self.all_feat_1[trial].shape[0])

    def compute_original_distmat(self):
        self.ori_distmat = []
        self.ori_rank = []
        self.ori_distmat_Q2Q = []
        self.ori_distmat_Q2G = []
        self.ori_distmat_G2Q = []
        self.ori_distmat_G2G = []
        self.ori_rank_Q2Q = []
        self.ori_rank_Q2G = []
        self.ori_rank_G2Q = []
        self.ori_rank_G2G = []
        self.ori_rank_QG2Q = []
        self.ori_rank_QG2G = []

        for trial in range(10):
            distmat_v = self._compute_ori_dist(self.all_feat_1[trial], self.all_feat_1[trial]) + \
                        self._compute_ori_dist(self.all_feat_2[trial], self.all_feat_2[trial]) + \
                        self._compute_ori_dist(self.all_feat_3[trial], self.all_feat_3[trial]) + \
                        self._compute_ori_dist(self.all_feat_4[trial], self.all_feat_4[trial]) + \
                        self._compute_ori_dist(self.all_feat_5[trial], self.all_feat_5[trial]) + \
                        self._compute_ori_dist(self.all_feat_6[trial], self.all_feat_6[trial])
            distmat_t = self._compute_ori_dist(self.all_feat_txt[trial], self.all_feat_txt[trial])
            distmat_joint = self._compute_ori_dist(self.all_feat_joint[trial], self.all_feat_joint[trial])

            if self.dataset in ('sysu', 'llcm'):
                original_dist = (distmat_v + distmat_t + distmat_joint) / 8.
            elif self.dataset == 'regdb':
                original_dist = (distmat_v + distmat_joint) / 7.

            q_num = self.q_num[trial]
            ori_distmat_Q2Q = original_dist.copy()[:q_num, :q_num]
            ori_distmat_Q2G = original_dist.copy()[:q_num, q_num:]
            ori_distmat_G2Q = original_dist.copy()[q_num:, :q_num]
            ori_distmat_G2G = original_dist.copy()[q_num:, q_num:]
            ori_distmat_Q2Q /= ori_distmat_Q2Q.max(axis=1)[:, np.newaxis]
            ori_distmat_Q2G /= ori_distmat_Q2G.max(axis=1)[:, np.newaxis]
            ori_distmat_G2Q /= ori_distmat_G2Q.max(axis=1)[:, np.newaxis]
            ori_distmat_G2G /= ori_distmat_G2G.max(axis=1)[:, np.newaxis]
            self.ori_distmat_Q2Q.append(ori_distmat_Q2Q)
            self.ori_distmat_Q2G.append(ori_distmat_Q2G)
            self.ori_distmat_G2Q.append(ori_distmat_G2Q)
            self.ori_distmat_G2G.append(ori_distmat_G2G)
            ori_rank_Q2Q = np.argsort(ori_distmat_Q2Q).astype(int)
            ori_rank_Q2G = np.argsort(ori_distmat_Q2G).astype(int)
            ori_rank_G2Q = np.argsort(ori_distmat_G2Q).astype(int)
            ori_rank_G2G = np.argsort(ori_distmat_G2G).astype(int)
            self.ori_rank_Q2Q.append(ori_rank_Q2Q)
            self.ori_rank_Q2G.append(ori_rank_Q2G)
            self.ori_rank_G2Q.append(ori_rank_G2Q)
            self.ori_rank_G2G.append(ori_rank_G2G)
            self.ori_rank_QG2Q.append(np.concatenate([ori_rank_Q2Q, ori_rank_G2Q], axis=0))
            self.ori_rank_QG2G.append(np.concatenate([ori_rank_Q2G, ori_rank_G2G], axis=0))

            original_dist /= original_dist.max(axis=1)[:, np.newaxis]
            original_rank = np.argsort(original_dist).astype(int)
            self.ori_distmat.append(original_dist)
            self.ori_rank.append(original_rank)

    def _get_k_reciprocal_index(self, query_index, rank_1, rank_2, k):
        forward_k_neighbor_index = rank_1[query_index, :k + 1]  # forward retrieval
        backward_k_neighbor_index = rank_2[forward_k_neighbor_index, :k + 1]  # backward retrieval
        k_reciprocal_row = np.where(backward_k_neighbor_index == query_index)[0]
        k_reciprocal_index = forward_k_neighbor_index[k_reciprocal_row]
        return k_reciprocal_index

    def _compute_jaccard_dist(self, features, q_num, g_num, all_num, fast=True):
        """
        - fast_version: fast calculation based on some tricks. It runs much faster, but harder to read.
        """
        jaccard_dist = np.zeros((q_num + g_num, q_num + g_num), dtype=np.float16)
        assert features.shape[0] == q_num + g_num
        assert features.shape[1] == all_num
        if fast:
            q_non_zero_index = [np.where(features[i, :] != 0)[0] for i in range(q_num + g_num)]
            g_non_zero_index = [np.where(features[:, j] != 0)[0] for j in range(all_num)]
            for i, query_feature in enumerate(features):
                minimum = np.zeros(q_num + g_num, dtype=np.float16)
                q_non_zero_index_i = q_non_zero_index[i]
                indices = [g_non_zero_index[idx] for idx in q_non_zero_index_i]
                for idx1, idx2 in zip(indices, q_non_zero_index_i):
                    minimum[idx1] += np.minimum(features[i, idx2], features[idx1, idx2])
                jaccard_dist[i] = 1 - minimum / (2 - minimum)
        else:
            for i, query_feature in enumerate(features):
                for j, gallery_feature in enumerate(features):
                    minimum = np.minimum(query_feature, gallery_feature).sum()
                    maximum = np.maximum(query_feature, gallery_feature).sum()
                    jaccard_dist[i, j] = 1 - minimum / maximum
        return jaccard_dist

    def _re_rank(self, original_dist, original_rank, k1, k2, q_num, g_num, all_num, trial, k3=None):
        """re-ranking"""
        '''1) k-reciprocal features'''
        k_reciprocal_features = np.zeros_like(original_dist, dtype=np.float16)  # i.e., `V` in paper
        for i in range(all_num):
            k_reciprocal_index = self._get_k_reciprocal_index(i, original_rank, original_rank, k=k1)

            if self.method == 'extended':
                if i < q_num:
                    extended_k_reciprocal_index = self._get_k_reciprocal_index(
                        i, self.ori_rank_Q2G[trial], self.ori_rank_G2Q[trial], k=k1,
                    ) + q_num
                else:
                    extended_k_reciprocal_index = self._get_k_reciprocal_index(
                        i - q_num, self.ori_rank_G2Q[trial], self.ori_rank_Q2G[trial], k=k1,
                    )
                k_reciprocal_index = np.unique(
                    np.append(k_reciprocal_index, extended_k_reciprocal_index)
                )

            k_reciprocal_incremental_index = k_reciprocal_index.copy()  # index after incrementally adding
            '''incrementally adding'''
            for j, candidate in enumerate(k_reciprocal_index):
                candidate_k_reciprocal_index = self._get_k_reciprocal_index(
                    candidate, original_rank, original_rank, k=round(k1 * self.alpha))
                if len(np.intersect1d(k_reciprocal_index, candidate_k_reciprocal_index)) \
                        > self.beta * len(candidate_k_reciprocal_index):
                    k_reciprocal_incremental_index = np.append(
                        k_reciprocal_incremental_index, candidate_k_reciprocal_index)
            k_reciprocal_incremental_index = np.unique(k_reciprocal_incremental_index)
            '''compute '''
            weight = np.exp(-original_dist[i, k_reciprocal_incremental_index])  # reassign weights with Gaussian kernel
            k_reciprocal_features[i, k_reciprocal_incremental_index] = weight / weight.sum()

        '''2) local query expansion'''
        if k2 != 1:
            k_reciprocal_expansion_features = np.zeros_like(k_reciprocal_features)
            for i in range(all_num):
                if k3 is None:
                    indices = original_rank[i, :k2]
                else:
                    assert k3 <= k2
                    indices = np.concatenate([
                        self.ori_rank_QG2Q[trial][i, :k2-k3],
                        self.ori_rank_QG2G[trial][i, :k3] + q_num,
                    ])
                k_reciprocal_expansion_features[i, :] = \
                    np.mean(k_reciprocal_features[indices, :], axis=0)
            k_reciprocal_features = k_reciprocal_expansion_features

        return k_reciprocal_features

    def _re_rank_constrained(self, ori_dist, ori_rank, k1, k2, q_num, g_num, all_num):
        """re-ranking"""
        dist_Q2G, dist_G2Q, dist_G2G = ori_dist[:q_num, q_num:], ori_dist[q_num:, :q_num], ori_dist[q_num:, q_num:]
        rank_Q2G, rank_G2Q, rank_G2G = ori_rank[:q_num, q_num:], ori_rank[q_num:, :q_num], ori_rank[q_num:, q_num:]

        '''1) k-reciprocal features'''
        k_reciprocal_features = np.zeros_like(ori_dist, dtype=np.float16)  # i.e., `V` in paper
        for i in range(all_num):
            if i < q_num:
                k_reciprocal_index = self._get_k_reciprocal_index(i, rank_Q2G, rank_G2Q, k=k1)
            else:
                k_reciprocal_index = self._get_k_reciprocal_index(i - q_num, rank_G2G, rank_G2G, k=k1)
            k_reciprocal_incremental_index = k_reciprocal_index.copy()  # index after incrementally adding
            '''incrementally adding'''
            for j, candidate in enumerate(k_reciprocal_index):
                candidate_k_reciprocal_index = self._get_k_reciprocal_index(
                    candidate, rank_G2G, rank_G2G, k=round(k1 * self.alpha)
                )
                if len(np.intersect1d(k_reciprocal_index, candidate_k_reciprocal_index)) \
                        > self.beta * len(candidate_k_reciprocal_index):
                    k_reciprocal_incremental_index = np.append(
                        k_reciprocal_incremental_index, candidate_k_reciprocal_index)
            k_reciprocal_incremental_index = np.unique(k_reciprocal_incremental_index)
            '''compute '''
            if i < q_num:  # reassign weights with Gaussian kernel
                weight = np.exp(-dist_Q2G[i, k_reciprocal_incremental_index])
            else:
                weight = np.exp(-dist_G2G[i - q_num, k_reciprocal_incremental_index])
            k_reciprocal_features[i, k_reciprocal_incremental_index] = weight / weight.sum()

        '''2) local query expansion'''
        if k2 != 1:
            k_reciprocal_expansion_features = np.zeros_like(k_reciprocal_features)
            for i in range(all_num):
                if i < q_num:
                    k_reciprocal_expansion_features[i, :] = \
                        np.mean(k_reciprocal_features[rank_Q2G[i, :k2], :], axis=0)
                else:
                    k_reciprocal_expansion_features[i, :] = \
                        np.mean(k_reciprocal_features[rank_G2G[i - q_num, :k2], :], axis=0)
            k_reciprocal_features = k_reciprocal_expansion_features

        return k_reciprocal_features

    def re_ranking(self, k1, k2, lam, trial, k3=None):
        q_num, g_num, all_num = self.q_num[trial], self.g_num[trial], self.all_num[trial]
        original_dist, original_rank = self.ori_distmat[trial], self.ori_rank[trial],
        if self.method in ('baseline', 'extended', 'divided', 'masked'):
            if self.method == 'divided':
                ori_dist = np.block([
                    [self.ori_distmat_Q2Q[trial], self.ori_distmat_Q2G[trial]],
                    [self.ori_distmat_G2Q[trial], self.ori_distmat_G2G[trial]],
                ])
                ori_rank = np.block([
                    [self.ori_rank_Q2Q[trial], self.ori_rank_Q2G[trial]],
                    [self.ori_rank_G2Q[trial], self.ori_rank_G2G[trial]],
                ])
            else:
                ori_dist = original_dist
                ori_rank = original_rank
            k_reciprocal_features = self._re_rank(
                ori_dist,
                ori_rank,
                k1, k2,
                q_num, g_num, all_num,
                trial,
                k3,
            )
            jaccard_dist = self._compute_jaccard_dist(k_reciprocal_features, q_num, g_num, all_num)
            return lam * original_dist[:q_num, q_num:] + (1 - lam) * jaccard_dist[:q_num, q_num:]
        elif self.method == 'constrained':
            '''constrained in gallery domain'''
            ori_dist = np.block([
                [self.ori_distmat_Q2Q[trial], self.ori_distmat_Q2G[trial]],
                [self.ori_distmat_G2Q[trial], self.ori_distmat_G2G[trial]],
            ])
            ori_rank = np.block([
                [self.ori_rank_Q2Q[trial], self.ori_rank_Q2G[trial]],
                [self.ori_rank_G2Q[trial], self.ori_rank_G2G[trial]],
            ])
            k_reciprocal_features_G = self._re_rank_constrained(ori_dist, ori_rank, k1, k2, q_num, g_num, all_num)
            jaccard_dist_G = self._compute_jaccard_dist(k_reciprocal_features_G, q_num, g_num, all_num)[:q_num, q_num:]
            '''constrained in query domain'''
            ori_dist = np.block([
                [self.ori_distmat_G2G[trial], self.ori_distmat_G2Q[trial]],
                [self.ori_distmat_Q2G[trial], self.ori_distmat_Q2Q[trial]],
            ])
            ori_rank = np.block([
                [self.ori_rank_G2G[trial], self.ori_rank_G2Q[trial]],
                [self.ori_rank_Q2G[trial], self.ori_rank_Q2Q[trial]],
            ])
            k_reciprocal_features_Q = self._re_rank_constrained(ori_dist, ori_rank, k1, k2, g_num, q_num, all_num)
            jaccard_dist_Q = self._compute_jaccard_dist(k_reciprocal_features_Q, g_num, q_num, all_num)[:g_num, g_num:]
            '''final distance'''
            jaccard_dist = (jaccard_dist_G + jaccard_dist_Q.T) / 2.
            return lam * original_dist[:q_num, q_num:] + (1 - lam) * jaccard_dist


if __name__ == '__main__':
    dataset = ['sysu', 'regdb', 'llcm'][0]
    method = ['baseline', 'constrained', 'extended', 'divided'][2]

    if dataset == 'sysu':
        FEAT_DIR = '/data1/dyh/results/Refer-VIReID/Git/SYSU_YYDS/all'
        # FEAT_DIR = '/data1/dyh/results/Refer-VIReID/Git/SYSU_YYDS/indoor'
    elif dataset == 'regdb':
        FEAT_DIR = '/data1/dyh/results/Refer-VIReID/Git/RegDB_YYDS'
    elif dataset == 'llcm':
        FEAT_DIR = '/data1/dyh/results/Refer-VIReID/Git/LLCM_YYDS'

    SEARCH_SPACE = {
        'k1': [35],
        'k2': [35],
        'lam': [0.1],
        'k3': [4],  # only for MA-LQE
    }
    for k, v in SEARCH_SPACE.items():
        print('============> {}: {} <==========='.format(k, v))

    DIST = Distmat(dataset, FEAT_DIR, method)

    results = []
    BEST_METRIC = -1
    BEST_CMC, BEST_MAP = None, None
    BEST_PARAM = None

    for params in tqdm(list(itertools.product(*SEARCH_SPACE.values()))):
        params = {k: v for k, v in zip(SEARCH_SPACE, params)}
        CMC, MAP = 0, 0

        if params['k2'] > params['k1']:
            continue

        for trial in range(10):
            distmat = DIST.re_ranking(**params, trial=trial)

            if dataset in ('sysu', 'llcm'):
                query_label = DIST.query_label[0]
                query_cam = DIST.query_cam[0]
                gallery_label = DIST.gallery_label[trial]
                gallery_cam = DIST.gallery_cam[trial]
                cmc, mAP, _ = eval('eval_'+dataset)(distmat, query_label, gallery_label, query_cam, gallery_cam)
            elif dataset == 'regdb':
                query_label = DIST.query_label[trial]
                gallery_label = DIST.gallery_label[trial]
                cmc, mAP, _ = eval_regdb(distmat, query_label, gallery_label)

            CMC += cmc
            MAP += mAP

        CMC /= 10
        MAP /= 10

        results.append([params, CMC[0], CMC[4], CMC[9], CMC[19], MAP])
        if CMC[0] + MAP > BEST_METRIC:
            BEST_CMC = CMC
            BEST_MAP = MAP
            BEST_PARAM = params
            BEST_METRIC = CMC[0] + MAP

    print(results)
    print('BEST_PARAMS: ', BEST_PARAM.items())
    print('BEST_METRICS: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}'.format(
        BEST_CMC[0], BEST_CMC[4], BEST_CMC[9], BEST_CMC[19], BEST_MAP))
    print_time('Done!')