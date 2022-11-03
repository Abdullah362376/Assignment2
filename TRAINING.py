from NN import *
alpha = 0.0001
iterations = 200

X = np.load("data.npy").T
Y = np.load("labels.npy")
Y = Y.reshape((1,141))
print("shape of data =",X.shape)
# print("shape of label =",Y.shape)


W1, b1, W2, b2, W3, b3, W4, b4, A4 = gradient_descent(X, Y, alpha, iterations) # you can comment these 8 lines if you  want
# to skip the trainig and show resuts
np.save("A4",A4)
np.save("F_W1f",W1)
np.save("F_b1f",b1)
np.save("F_W2f",W2)
np.save("F_b2f",b2)
np.save("F_W3f",W3)
np.save("F_b3f",b3) # comment up untill here if you want to skip training
np.save("F_W4f",W4)
np.save("F_b4f",b4)
print (A4)
W1 = np.load("F_W1f.npy")
b1 = np.load("F_b1f.npy")
W2 = np.load("F_W2f.npy")
b2 = np.load("F_b2f.npy")
W3 = np.load("F_W3f.npy")
b3 = np.load("F_b3f.npy")
W4 = np.load("F_W4f.npy")
b4 = np.load("F_b4f.npy")

Z1, A1, Z2, A2 ,Z3 ,A3 ,Z4 ,A4 = forward_propagation(W1, b1, W2, b2, W3, b3,W4,b4, X)
predictions = get_predictions(A4)
accuracy = get_accuracy(predictions, Y)
print("Accuracy = ",accuracy,"%")
f1= analysis(Y)