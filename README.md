# BootEA
Bootstrapping Entity Alignment with Knowledge Graph Embedding

## Code

### Dependencies
* Python 3
* Tensorflow 1.x 
* Scipy
* Numpy

### Parameter settings
* mu_1: a bigger mu_1 (such as 1.0) would lead to a better results measured by Hits@k. However, the bigger value would hinder the performance measured by MR (mean rank). Given the above, a small mu_1 (such as 0.2) would be a good choice.
