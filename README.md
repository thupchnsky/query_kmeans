# K-means clustering with side information (Query K-means)
Implementation of query k-means algorithm [(paper)](https://proceedings.neurips.cc/paper/2018/file/0655f117444fc1911ab9c6f6b0139051-Paper.pdf).

# Example
Our method supports both noiseless and noisy queries. For noisy queries, we use Algorithm 2 in [CLUSTERING WITH NOISY QUERIES](https://arxiv.org/pdf/1706.07510.pdf) by Mazumdar and Saha as a subprocedure.

Here are two examples for synthetic Gaussian mixture data. Detailed description of each input can be found in the code.

**Noiseless**
```
python noiseless.py --k=3 --d=3 --balanced
```

**Noisy**
```
python noisy.py --k=3 --d=3 --size=15000 --balanced
```

# Contact
Please contact Chao Pan (chaopan2@illinois.edu) if you have any question.

# Citation
If you find our code or work useful, please consider citing our paper:
```
@article{chien2018query,
  title={Query k-means clustering and the double dixie cup problem},
  author={Chien, I and Pan, Chao and Milenkovic, Olgica},
  journal={Advances in Neural Information Processing Systems},
  volume={31},
  year={2018}
}
```
