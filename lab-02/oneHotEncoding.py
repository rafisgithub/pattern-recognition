
import pandas as pd

dataset = pd.read_csv("./content/CategoryDataset.csv")
# print(dataset)

import numpy as np




from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
x = dataset.iloc[:,0:3].values
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0,1,2])],remainder='passthrough')
x = np.array(ct.fit_transform(x))

print(x)