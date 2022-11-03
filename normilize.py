import numpy as np
import matplotlib as plt
Z = np.load("data_set.npy")
X = np.load("data_set.npy")  # load the data set
Y = np.load("labels.npy")    # load the label se


X -= np.mean(X,axis=0)

X /= np.std(X,axis=0)

np.save("data.npy",X)
np.save("labels.npy",Y)

print(np.mean(X))
print(np.std(X))


# print(X.size,X.shape,X.ndim)
# print(X1.size,X1.shape,X1.ndim)
print(np.allclose(X,Z))






