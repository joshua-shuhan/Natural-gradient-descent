import numpy as np
import pandas as pd
from scipy import linalg

data = pd.read_csv('IRIS.csv')
data.loc[data['species'] == 'Iris-setosa', 'species'] = 0
data.loc[data['species'] == 'Iris-versicolor', 'species'] = 1
data.loc[data['species'] == 'Iris-virginica', 'species'] = 2

data = data.iloc[np.random.permutation(len(data))]  # Shuffle the data
x_train = data.iloc[0:100, 0:4].to_numpy().astype(np.double)
y_train = data.iloc[0:100, 4].to_numpy().astype(int)

x_test = data.iloc[100:, 0:4].to_numpy().astype(np.double)
y_test = data.iloc[100:, 4].to_numpy().astype(int)

