#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import argparse
from utils import *
from collections import Counter
from sklearn.cluster import KMeans


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query K-means [Noisy]')
    parser.add_argument('--k', type=int, default=2, help='Number of clusters')
    parser.add_argument('--d', type=int, default=2, help='Data dimension')
    parser.add_argument('--size', type=int, default=15000, help='Size of the largest cluster')
    parser.add_argument('--eps', type=float, default=0.2, help='Parameter epsilon')
    parser.add_argument('--delta', type=float, default=0.2, help='Parameter delta')
    parser.add_argument('--pe', type=float, default=0.01, help='Parameter pe')
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
    # noisy(data, labels, alpha, args.k, args.pe)
    M, query_centers, query_comp = noisy(data, labels, alpha, args.k, args.pe)
    query_loss = clustering_loss(data, query_centers)
    print('Optimal objective:', opt_loss)
    print('(1 + eps) * optimal objective:', (1 + args.eps) * opt_loss)
    print('Query K-means objective:', query_loss)
    ub = 64 ** 2 * M * args.k ** 2 * np.log(M) / (1 - 2 * args.pe) ** 4
    # show log value in case it is huge
    print('Log query complexity upper bound:', np.log(ub))
    print('Log real query complexity:', np.log(query_comp))



