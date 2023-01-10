#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import random
from tqdm import tqdm


def generateGaussianData(mu, sigma=None, count=1000):
    '''
    :param mu: list, mean
    :param sigma: array, covariance matrix
    :param count: int, number of points
    :return: data: array, generated multivariate normal distribution
    '''
    d = len(mu)
    if sigma is None:
        sigma = np.eye(d) * 0.5
    else:
        assert sigma.shape[0] == d and sigma.shape[1] == d
    data = np.random.multivariate_normal(mu, sigma, count)
    return data


def noiseless(data, labels, k, lb):
    '''
    :param data: array, input data
    :param labels: array, true labels
    :param k: int, number of clusters
    :return:
        query_mean: array, estimated centers
        query_complexity: int, number of queries
    '''
    clusters = [[] for _ in range(k)]
    clusters_sizes = [0] * k
    n = data.shape[0]
    d = data.shape[1]
    query_complexity = 0
    while min(clusters_sizes) < lb:
        idx = np.random.randint(n)
        idx_label = labels[idx]
        query_complexity += idx_label + 1
        clusters[idx_label].append(data[idx].reshape(1, d))
        clusters_sizes[idx_label] += 1
    query_mean = np.zeros((k, d))
    for i in range(k):
        query_mean[i] = np.mean(np.concatenate(clusters[i], axis=0), axis=0)
    return query_mean, query_complexity


def noisy(data, labels, alpha, k, pe):
    '''
    An implementation of Algorithm 2 by Mazumdar and Saha [24].

    :param data: array, input data
    :param labels: array, true labels
    :param alpha: float, size imbalance parameter
    :param k: int, number of clusters
    :param pe: float, error probability
    :return:
        M: int, size of sampled data points
        query_mean: array, estimated centers
        query_complexity: int, number of queries
    '''
    tmp = 128 * alpha * k ** 2 / (1 - 2 * pe) ** 4
    M = int(tmp * np.log(tmp))
    # sample without replacement
    node_list = np.random.permutation(data.shape[0])[0: M]

    # noisy clustering algorithm
    active_clusters = [set() for _ in range(k)]
    tmp_clusters = [set() for _ in range(k)]
    unassigned_nodes = list(node_list)
    N = int(64 * k ** 2 * np.log(M) / (1 - 2 * pe) ** 4)
    c = 16 / (1 - 2 * pe) ** 2
    adj_list = {}
    query_complexity = 0

    while len(unassigned_nodes) > 0:
        print('### New iteration starts ###')

        # Phase 1
        print('## Phase 1 ##')
        random.shuffle(unassigned_nodes)
        new_nodes_needed = N - len(list(adj_list.keys()))
        if len(unassigned_nodes) <= new_nodes_needed:
            tmp_nodes = unassigned_nodes.copy()
            unassigned_nodes = []
        else:
            tmp_nodes = list(unassigned_nodes[0: new_nodes_needed])
            unassigned_nodes = unassigned_nodes[new_nodes_needed:]

        ori_adj_list_keys = list(adj_list.keys())
        # record positive edges for tmp nodes
        for new_node_idx in tmp_nodes:
            adj_list[new_node_idx] = set()
        adj_list_keys = list(adj_list.keys())

        # update the edges
        # edges with previous round nodes
        for new_node_idx in tqdm(tmp_nodes):
            for node_idx in ori_adj_list_keys:
                if new_node_idx != node_idx:
                    query_complexity += 1
                    flag1 = labels[new_node_idx] == labels[node_idx] and random.random() < 1 - pe
                    flag2 = labels[new_node_idx] != labels[node_idx] and random.random() < pe
                    if flag1 or flag2:
                        adj_list[new_node_idx].add(node_idx)
                        adj_list[node_idx].add(new_node_idx)

        # edges with new nodes
        for i in tqdm(range(len(tmp_nodes))):
            for j in range(i+1, len(tmp_nodes)):
                query_complexity += 1
                flag1 = labels[tmp_nodes[i]] == labels[tmp_nodes[j]] and random.random() < 1 - pe
                flag2 = labels[tmp_nodes[i]] != labels[tmp_nodes[j]] and random.random() < pe
                if flag1 or flag2:
                    adj_list[tmp_nodes[i]].add(tmp_nodes[j])
                    adj_list[tmp_nodes[j]].add(tmp_nodes[i])

        # Phase 2
        print('## Phase 2 ##')
        val_T = getT(pe, len(adj_list_keys), N * np.log(M))
        val_theta = getTheta(pe, len(adj_list_keys), N * np.log(M))
        for i in tqdm(range(len(adj_list_keys))):
            for j in range(i+1, len(adj_list_keys)):
                # exact set difference is slow
                # uncomment for exact set difference
                # neighbor_i = adj_list[adj_list_keys[i]]
                # neighbor_j = adj_list[adj_list_keys[j]]
                # if len(neighbor_i) >= val_T and len(neighbor_j) >= val_T and \
                #     len(neighbor_i - neighbor_j) + len(neighbor_j - neighbor_i) <= val_theta:
                #     tmp_clusters[labels[adj_list_keys[i]]].add(adj_list_keys[i])
                #     tmp_clusters[labels[adj_list_keys[i]]].add(adj_list_keys[j])

                # use approximate set difference to accelerate this procedure
                # if u and v belong to the same cluster, then with prob at least 1-2/n^2 they would satisfy those two conditions
                prob = 2 / M ** 2
                flag1 = labels[adj_list_keys[i]] == labels[adj_list_keys[j]] and random.random() < 1 - prob
                flag2 = labels[adj_list_keys[i]] != labels[adj_list_keys[j]] and random.random() < prob
                if flag1 or flag2:
                    tmp_clusters[labels[adj_list_keys[i]]].add(adj_list_keys[i])
                    tmp_clusters[labels[adj_list_keys[i]]].add(adj_list_keys[j])

        # move large enough tmp_clusters to active clusters
        for i in range(k):
            if len(tmp_clusters[i]) >= N / k:
                active_clusters[i] = active_clusters[i].union(tmp_clusters[i])
                # remove them from the graph
                for remove_idx in tmp_clusters[i]:
                    for node_idx in adj_list.keys():
                        adj_list[node_idx].discard(remove_idx)
                    if remove_idx in adj_list.keys():
                        del adj_list[remove_idx]
                tmp_clusters[i] = set()

        # Phase 3
        print('## Phase 3 ##')
        for idx in tqdm(unassigned_nodes):
            for i in range(k):
                if len(active_clusters[i]) >= c * np.log(M):
                    pos_count = 0
                    total_count = 0
                    for assigned_node in active_clusters[i]:
                        total_count += 1
                        query_complexity += 1
                        flag1 = labels[assigned_node] == labels[idx] and random.random() < 1 - pe
                        flag2 = labels[assigned_node] != labels[idx] and random.random() < pe
                        if flag1 or flag2:
                            pos_count += 1
                        if total_count >= c * np.log(M):
                            break
                    if pos_count >= total_count / 2:
                        active_clusters[i].add(idx)
                        unassigned_nodes.remove(idx)
                        break

        # print(len(active_clusters[0]), len(active_clusters[1]))
        # check if all active cluster sizes are greater than lower bound
        sizes = list(map(len, active_clusters))
        if min(sizes) >= N / k:
            break

    # now active clusters record the points for each cluster
    clusters = [[] for _ in range(k)]
    d = data.shape[1]
    for i in range(k):
        for idx in active_clusters[i]:
            clusters[i].append(data[idx].reshape(1, d))
    query_mean = np.zeros((k, d))
    for i in range(k):
        query_mean[i] = np.mean(np.concatenate(clusters[i], axis=0), axis=0)
    return M, query_mean, query_complexity


def getT(pe, a, Nlogn):
    '''
    :param pe: float, error probability
    :param a: int, input of the function
    :param Nlogn: float, N * logn
    :return: value of the T function
    '''
    return pe * a + 6 / (1 - 2 * pe) * np.sqrt(Nlogn)


def getTheta(pe, a, Nlogn):
    '''
    :param pe: float, error probability
    :param a: int, input of the function
    :param Nlogn: float, N * logn
    :return: value of the theta function
    '''
    return 2 * pe * (1 - pe) * a + 2 * np.sqrt(Nlogn)


def clustering_loss(data, centroids):
    '''
    :param data: array, input data
    :param centroids: array, centers
    :return: loss: float, computed loss
    '''
    loss = 0.0
    for i_data in range(data.shape[0]):
        d = np.linalg.norm(data[i_data, :] - centroids, axis=1)
        loss += np.min(d) ** 2
    return loss
