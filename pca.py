import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


### Exploration
df = pd.read_csv("wine.csv")
print("Number of null values = {}".format(df.isnull().any().sum()))

y = df.values[:, 0]
X = df.values[:, 1:]
numSamples = len(y)
numClasses = len(np.unique(y))
frequency = {i:np.sum(y == i) for i in y}
percentage = {i:np.sum(y == i)/y.shape[0]*100 for i in y}
print("Number of samples = {}".format(numSamples))
print("Number of classes = {}".format(numClasses))
print("Class frequency = {}".format(frequency))
print("Class percentage = {}".format(percentage))


### Split data between test and training
X1, X2, y1, y2 = train_test_split(X, y, test_size = .33, random_state=1 )

## Normalize values between 0 and 1 
scaler = MinMaxScaler()
X1norm = scaler.fit_transform(X1)
X2norm = scaler.fit_transform(X2)

### Choosing values for hidden layer using Principle Component Analysis
pca = PCA(n_components=.99)
pca.fit(X1norm)
reducedPCA = pca.transform(X1norm)
numFeaturesPCA = reducedPCA.shape
print("Number of features after PCA = {}".format(numFeaturesPCA))









