import pandas as pd
import numpy as np
from sklearn import preprocessing
import datetime
from dateutil.parser import parse
from numpy.random import seed
from numpy.random import randn

iso_encoder = preprocessing.LabelEncoder()
dataframe = pd.read_csv('covid-data.csv')
# print(dataframe)

df = dataframe[(dataframe.continent).notna()]
# print(df)
df['iso_code'] = iso_encoder.fit_transform(df['iso_code'])
iso_encoder2 = preprocessing.LabelEncoder()
df['continent'] = iso_encoder2.fit_transform(df['continent'])
# NEEDS TO BE UNCOMMENTED LATER
for (index,datetype) in enumerate(df.date):
    some_date = parse(str(datetype))
    mydate = datetime.datetime(2019, 1, 1)
    the_days = some_date - mydate
    df.date[index] = the_days.days
df = df.iloc[:, 0:8]
df = df.dropna()
df.reset_index(drop=True)
global final_df
final_df = pd.DataFrame(columns=['iso_code', 'continent','location', 'date',  'total_cases'
                                 ,'new_cases', 'total_deaths' , 'new_deaths'  ,'previous_day_cases'  ,'previous_day_deaths'])
for region, df_region in df.groupby('iso_code'):
    df_region['previous_day_cases'] = df_region.new_cases.shift(1)
    df_region['previous_day_deaths'] = df_region.new_deaths.shift(1)
    final_df = final_df.append(df_region)
    print(df_region)

print (final_df)
# print(df)
# print(df)
# m = 2
# n = 10
#
# weights = np.zeros((int(m),int(n)))
# # print(weights)
#
# X = [[]]
# X[0] = df.iso_code.to_numpy()
# X.append(df.continent.to_numpy())
# weights = []
# weights.append(X)
#
# for i in range(0,9):
#     weights.append(X)
#
# seed(7)
# # print(weights)
# i = len(weights)
# j = len(weights[0])
# k = len(weights[0][0])
# print(values[0])
X = []
X.append(final_df.iso_code.to_numpy())
X.append(final_df.continent.to_numpy())
# print(X)
seed(7)
W = np.random.randn(10,2)
# print(W)
O = np.matmul(W,X)

print("###############It is result time!##################")
# print(O)

new_input_layer = O.tolist()
new_input_layer.append(final_df.date.to_numpy())
new_input_layer.append(final_df.previous_day_cases.to_numpy())
new_input_layer.append(final_df.previous_day_deaths.to_numpy())

print(new_input_layer)
print (len(new_input_layer))
print(len(new_input_layer[0]))

# now we can use new_input_layer as X for training based on below functions
X = np.asarray(new_input_layer)
y1 = final_df.iloc[:,5].to_numpy()
y2 = final_df.iloc[:,7].to_numpy()
y = []
y.append(y1)
y.append(y2)
y = np.asarray(y)
print(y)

def sigmoid(t):
    if (t==0.0):
        return 1/(1+1)

    return 1 / (1 + np.exp(-t))


# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)


# Class definition
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)  # considering we have 4 nodes in the hidden layer
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output),
                                                 self.weights2.T) * sigmoid_derivative(self.layer1))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()


NN = NeuralNetwork(X, y)
for i in range(1500):  # trains the NN 1,000 times
    if i % 100 == 0:
        print("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(X))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str(NN.feedforward()))
        print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss
        print("\n")

    NN.train(X, y)



