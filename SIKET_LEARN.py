import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

x_train = np.load("data.npy")
y_train = np.load("labels.npy")
nn=MLPClassifier(activation='logistic',solver='sgd',hidden_layer_sizes=(40,1),random_state=1)

print(nn.fit(x_train,y_train))