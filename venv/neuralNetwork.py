import pandas as pd
import numpy as np
from sklearn import preprocessing
import datetime
from dateutil.parser import parse
from numpy.random import seed
from numpy.random import randn

iso_encoder = preprocessing.LabelEncoder()
dataframe = pd.read_csv('covid-data.csv')
print(dataframe)

df = dataframe[(dataframe.continent).notna()]
print(df)
df['iso_code'] = iso_encoder.fit_transform(df['iso_code'])
iso_encoder2 = preprocessing.LabelEncoder()
df['continent'] = iso_encoder2.fit_transform(df['continent'])
# NEEDS TO BE UNCOMMENTED LATER
# for (index,datetype) in enumerate(df.date):
#     datetime = parse(str(datetype))
#     df.date[index] = datetime

# print(df)
# m = 2
# n = 10
#
# weights = np.zeros((int(m),int(n)))
# print(weights)

X = [[]]
X[0] = df.iso_code.to_numpy()
X.append(df.continent.to_numpy())
weights = [[[]]]
weights[0] = X

for i in range(0,9):
    weights.append(X)

seed(7)
# print(weights)
i = len(weights)
j = len(weights[0])
k = len(weights[0][0])
# print(values[0])
W = np.random.randn(i,j,k)
print(W)
global O
O = []
for i in range(0,len(W))

print("###############It is result time!##################")
print(O)
# print(dataframe.iso_code)


