# encoding=utf-8
import pandas as pd
import numpy as np
from sklearn import datasets, decomposition

X = pd.read_csv('../DATA/dataset.csv')
pca = decomposition.PCA(n_components=128)  # 使用默认的 n_components
# pca.fit(X)
X_r = pca.fit_transform(X)
print(X_r[:3])  # 默认n_components 没有降维,特征是按奇异值从大到小后排序后的
# print('explained variance ratio : %s' % str(pca.explained_variance_ratio_))


dataframe_new = pd.DataFrame(X_r)
dataframe_new.to_csv('../DATA/data_compl.csv', index=False, header=False)
