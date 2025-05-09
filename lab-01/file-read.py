import pandas as pd

dataset = pd.read_csv("./data/OnlineShoppingStatus.csv")
# print(dataset)

# dataset = dataset.iloc[:,0:3]

# print(dataset)

features = dataset.iloc[:,0:-1].values
# print(features)

labels = dataset.iloc[:,-1].values

# print(labels)

from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(missing_values=np.nan,strategy='mean')

imputer.fit(features[:,1:-1])

features[:,1:-1] = imputer.transform(features[:,1:-1])

print(features)
