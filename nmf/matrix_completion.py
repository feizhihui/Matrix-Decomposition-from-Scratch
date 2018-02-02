# encoding=utf-8

import numpy as np
import pandas as pd
import time
from sklearn.decomposition import NMF

starttime = time.time()

dataframe = pd.read_csv('../DATA/dataset.csv')
X = dataframe.values

model = NMF(n_components=5, init='random', random_state=0)
W = model.fit_transform(X)
H = model.components_

endtime = time.time()

X_ = W @ H

dataframe_new = pd.DataFrame(X_, columns=dataframe.columns)
dataframe_new[dataframe != 0] = dataframe[dataframe != 0]

dataframe_new.to_csv('../DATA/data_compl.csv', index=False)

print((endtime - starttime) / 60, 'minutes')
