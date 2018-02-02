# encoding=utf-8
import pandas as pd
import numpy as np

n_components = 5


def eig_transform(sigma):  # return a matrix
    a = np.eye(len(sigma), dtype=np.float32) * sigma
    return a


dataframe = pd.read_csv('../DATA/dataset.csv')
X = dataframe.values

mean = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
X_norm = X - mean

u, sigma, v = np.linalg.svd(X_norm)

eig = eig_transform(sigma[:n_components])
X_new = u[:, :n_components] @ eig
print(X_new[:3])

# eigs, vec = np.linalg.eig(X @ X.T)
# print(eigs[:5])
