import numpy as np

def initialize_parameters():# initialize_parameters() : this is a function that will initilize the wight and bias parameter necessary for the
#neural network , this function does not take any argument , althoug you can set the value of the nudes per hidden layer
# by changing n=10 to any other nudal number.
    Din = 4096
    n  = 500
    W1 = np.random.rand(n, 4096) / np.sqrt(2/Din)
    b1 = np.zeros((n, 1))
    W2 = np.random.rand(n, n) / np.sqrt(2/n)
    b2 = np.zeros((n, 1))
    W3 = np.random.rand(n, n) / np.sqrt(2/n)
    b3 = np.zeros((n, 1))
    W4 = np.random.rand(1, n) / np.sqrt(2/n)
    b4 = np.zeros((1, 1))
    np.save("W1", W1)
    np.save("b1", b1)
    np.save("W2", W2)
    np.save("b2", b2)
    np.save("W3", W3)
    np.save("b3", b3)
    np.save("W4", W4)
    np.save("b4", b4)
    print (W1.shape,W2.shape,W3.shape,W4.shape,b1.shape,b2.shape,b3.shape,b4.shape)
    return W1, b1, W2, b2, W3, b3, W4, b4

def ReLU(Z):# 3- ReLU(Z) : this is an activation function (rectified linear activation unit) , it is one of the many recomended activation function there is
# it is very simple actully , it outputs a zero when the input is negative , and outputs a Z it self if the input is positive
    return np.maximum(Z, 0)

def ReLU_deriv(Z): # 6-ReLU_deriv(Z): this is a deravitive of the activation function of the hedden layer , we need the deravative for the
# back probagation , the deravative is very easy of you look at the basic graph of the ReLU function , you will relize
# that when Z is negative then the dervation is 0 , and when it is pozative it is 1
    return Z > 0

def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    return t

def sigmoid(Z): # 4-sigmoid(Z) : sigmoid is also an activation fuction , It is usually i,plementd to activate the output layer of the
# neural network , especially in a binary classification neural network \
    return 1 / (1 + np.exp(-Z))
def Batch_norm(A):
    A -= np.mean(A, axis=0)

    A /= np.std(A, axis=0)
    return A

def forward_propagation(W1, b1, W2, b2, W3, b3,W4,b4, X): # 5-forward_propagation(W1, b1, W2, b2, W3, b3, X):
# it is the first functionn of the neural network logic , it propagate the input features through the nudes by the help
# of the parameter we initilized to find the out put nude acivation output , A3 in this case
    A0 = X
    Z1 = np.dot(W1, A0) + b1
    A1 = ReLU(Z1)
    # A1 = Batch_norm(A1)
    Z2 = np.dot(W2, A1) + b2
    A2 = ReLU(Z2)
    # A2 = Batch_norm(A2)
    Z3 = np.dot(W3, A2) +b3
    A3 = ReLU(Z3)
    # A3 = Batch_norm(A3)
    Z4 = np.dot(W4, A3) +b4
    A4 = sigmoid(Z4)
    return Z1, A1, Z2, A2 ,Z3 ,A3, Z4, A4

def backward_propagation(Z1, A1, Z2, A2 ,Z3 ,A3,Z4,A4,W2,W3,W4,X,Y): # 7-backward_propagation(Z1, A1, Z2, A2 ,Z3 ,A3,W2,W3,X,Y):
# this is backward probagation , it is the seconed step of NN , it basically find out how much the parameters are off
# from the actual labels. this step is necessary in updating the parameters.
    m   = Y.size
    dZ4 = A4-Y
    dW4 = (1/ m) * dZ4.dot(A3.T)
    db4 = (1/ m) * np.sum(dZ4)
    dZ3 = W4.T.dot(dZ4) * ReLU_deriv(Z3)
    dW3 = (1/ m) * dZ3.dot(A2.T)
    db3 = (1/ m) * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = (1/ m) * dZ2.dot(A1.T)
    db2 = (1/ m) * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = (1/ m) * dZ1.dot(X.T)
    db1 = (1/ m) * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3, dW4 , db4

def Tune_Parameters(W1, b1, W2, b2, W3, b3,W4,b4, dW1, db1, dW2, db2, dW3, db3,dW4,db4 , alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    W4 = W4 - alpha * dW4
    b4 = b4 - alpha * db4

    return W1, b1, W2, b2, W3, b3, W4, b4

# 9- get_predictions(A3): this is a very important function , it is basically inputs the output of the last layer
#activation function , and see if it larger than the threashhold , it returns 1, otherwise it returns 0 through the logical operation #
# (A3 > threshold) which return either True or false
def get_predictions(A4):
    threshold = 0.5
    predictions = (A4 > threshold)
    np.save("training_predictions",predictions)
    return predictions

def get_accuracy(predictions, Y):# 10- get_accuracy(predictions, Y): this is a function that gives the accuracy , basically , it sum up all the matching answers between your
# prediction and the lables , then devide it over the lable length
    return (np.sum(predictions == Y) / Y.size) *100

def compute_cost(A4, Y): #11- compute_cost(A3, Y): this function represent the error of your predection

    m = Y.shape[1]  # number of example

    # Compute the cross-entropy cost
    logprobs = np.dot(Y, np.log(A4).T) + np.dot((1 - Y), np.log((1 - A4)).T)
    cost = -logprobs / m

    cost = float(np.squeeze(cost))  # makes sure cost is the dimension we expect. E.g., turns [[17]] into 17
    assert (isinstance(cost, float))

    return cost

def gradient_descent(X, Y, alpha, iterations): # 12-gradient_descent(X, Y, alpha, iterations): this is the gradient decent
    W1, b1, W2, b2, W3, b3, W4, b4 = initialize_parameters()
    W1 = np.load("W1.npy")
    b1 = np.load("b1.npy")
    W2 = np.load("W2.npy")
    b2 = np.load("b2.npy")
    W3 = np.load("W3.npy")
    b3 = np.load("b3.npy")
    W4 = np.load("W4.npy")
    b4 = np.load("b4.npy")
    cost_list = []
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3, Z4, A4 = forward_propagation(W1, b1, W2, b2, W3, b3, W4, b4, X)
        cost = compute_cost(A4,Y)
        cost_list.append(cost)
        dW1, db1, dW2, db2, dW3, db3, dW4, db4 = backward_propagation(Z1, A1, Z2, A2 ,Z3 , A3, Z4, A4, W2, W3,W4, X, Y)
        W1, b1, W2, b2, W3, b3, W4, b4 = Tune_Parameters(W1, b1, W2, b2, W3, b3,W4,b4, dW1, db1, dW2, db2, dW3, db3,dW4,db4 , alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            print("cost      =", cost)
            # predictions = get_predictions(A3)
            # print("accuraccy =",get_accuracy(predictions, Y))
    np.save("cost_listK", cost_list)
    return W1, b1, W2, b2, W3, b3, W4, b4, A4


def analysis(Y): # 13- analysis(): this is a function to determine the confuesion matrix basically
    print("Here is the analysis :")
    predected = np.load("training_predictions.npy")
    predected = predected[0]
    label = Y[0]
    true_negative = 0
    true_positve = 0
    false_negative = 0
    false_positive = 0
    total_ones=0
    total_zeros=0
    for (f, b) in zip(label, predected):

        if (f == b and f == 0):
            total_zeros += 1
            true_negative += 1
        elif (f == b and f == 1):
            total_ones+=1
            true_positve += 1
        elif (f != b and b == 1):
            false_positive += 1
        elif (f != b and b == 0):
            false_negative += 1
    print("true positive  ", true_positve)
    print("true negative  ", true_negative)
    print("False positve  ", false_positive)
    print("false negative ", false_negative)
    Recall = (true_positve / (true_positve + false_negative))
    Precision = (true_positve / (true_positve + false_positive))
    F1_Score = 2*((Precision*Recall)/(Precision+Recall))
    print("RECALL     = ",Recall *100,"%")
    print("PRECISION  = ",Precision *100,"%")
    print("*F1_SCORE* = ",F1_Score *100,"%")