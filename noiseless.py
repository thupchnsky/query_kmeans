#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import argparse
from utils import *
from collections import Counter
from sklearn.cluster import KMeans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query K-means [Noiseless]')
    parser.add_argument('--k', type=int, default=5, help='Number of clusters')
    parser.add_argument('--d', type=int, default=5, help='Data dimension')
    parser.add_argument('--size', type=int, default=1000, help='Size of the largest cluster')
    parser.add_argument('--eps', type=float, default=0.1, help='Parameter epsilon')
    parser.add_argument('--delta', type=float, default=0.1, help='Parameter delta')
    parser.add_argument('--balanced', action='store_true', default=False, help='Generate equal-size clusters')
    args = parser.parse_args()

    print('### Data Generation ###')
    data = []
    # generate Gaussian data on a simplex, fix d = k for simplicity here
    assert args.d >= args.k
    for i in range(args.k):
        mu = [0] * args.d
        mu[i] = 1
        if args.balanced:
            data.append(generateGaussianData(mu=mu, count=args.size))
        else:
            data.append(generateGaussianData(mu=mu, count=args.size // (i + 1)))
    data = np.concatenate(data, axis=0)
    print('Shape of synthetic data:', data.shape)

    # obtain the *approximate* true labels and optimal objective from Kmeans
    kmeans = KMeans(n_clusters=args.k, n_init=10).fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    opt_loss = clustering_loss(data, centers)

    # cluster size imbalance alpha depends on labels
    sizes = Counter(labels)
    alpha = data.shape[0] / args.k / min(sizes.values())
    print('Cluster size imbalance alpha:', alpha)

    # query kmeans
    print('### Query K-means ###')
    m = args.k / args.delta / args.eps
    query_centers, query_comp = noiseless(data, labels, args.k, m)
    query_loss = clustering_loss(data, query_centers)
    print('Optimal objective:', opt_loss)
    print('(1 + eps) * optimal objective:', (1 + args.eps) * opt_loss)
    print('Query K-means objective:', query_loss)
    ub = alpha * args.k ** 3 / args.eps / args.delta
    print('Query complexity upper bound:', ub)
    print('Real query complexity:', query_comp)



