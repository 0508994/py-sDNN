import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return np.exp(-x) / ((1 + np.exp(-x))**2)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1. - x * x

def relu(x):
    return x * (x > 0)

def drelu(x):
    return 1. * (x > 0)

def softmax0(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

def dsoftmax0(x): 
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x. 
    # s.shape = (1, n) 
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    s = x.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)

def softmax(X):
    exps = np.exp(X)
    return exps / np.sum(exps, axis=1, keepdims=True)


if __name__ == '__main__':
    a = np.array([[0.1, 0.2]])
    print(softmax(a))
    print(dsoftmax(a))