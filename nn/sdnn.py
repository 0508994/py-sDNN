import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import style

np.seterr(all='ignore')
style.use('fivethirtyeight')

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

# def softmax(z):
#     assert len(z.shape) == 2
#     s = np.max(z, axis=1)
#     s = s[:, np.newaxis]
#     e_x = np.exp(z - s)
#     div = np.sum(e_x, axis=1)
#     div = div[:, np.newaxis]
#     return e_x / div

def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

class DNN:
    def __init__(self, shape, debug=False):
        assert len(shape) > 2
        self.shape = shape
        self.Ws = []
        self.bs = []

        for i in range(1, len(shape)):
            if debug:
                self.Ws.append(np.ones([shape[i - 1], shape[i]])) 
                self.bs.append(np.zeros(shape[i])) 
            else:
                self.Ws.append(np.random.randn(shape[i - 1], shape[i]) * np.sqrt(2.0 / shape[i - 1]))
                self.bs.append(np.zeros(shape[i])) #* np.sqrt(2.0 / shape[i - 1]))
                

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def _forward(self, X):
        self.zs = []    # dot products between layers
        self.acts = [X] # activation function applied on dot product between layers
        tmp = X
        for i in range(len(self.Ws)):
            tmp = np.dot(tmp, self.Ws[i]) + self.bs[i]
            self.zs.append(tmp)
            if i != len(self.Ws) - 1:
                tmp = relu(tmp)
                self.acts.append(tmp)
        self.y_hat = sigmoid(tmp)

    def square_error(self, X, y):
        self._forward(X)
        return 0.5 * np.sum((y - self.y_hat)**2)

    def compute_grads(self, X, y):
        #self._forward(X)
        delta = np.multiply(-(y - self.y_hat), dsigmoid(self.zs[-1]))
        dJdW = np.dot(self.acts[-1].T, delta)
        dJdb = delta[0] # hope this one just werks :}

        self.zs.pop()                        
        self.acts.pop()
        
        grads_w = [dJdW]
        grads_b = [dJdb]

        for w, z, a in zip(reversed(self.Ws), reversed(self.zs), reversed(self.acts)):
            delta = np.dot(delta, w.T) * drelu(z)
            dJdW = np.dot(a.T, delta)
            dJdb = delta[0]
            grads_w.insert(0, dJdW)
            grads_b.insert(0, dJdb)    

        return np.concatenate([*grads_w, *grads_b], axis=None)

    def compute_num_grads(self, X, y):
        params_init = self.get_params()
        numgrad = np.zeros(params_init.shape)
        perturb = np.zeros(params_init.shape)
        e = 1e-4

        for p in range(len(params_init)):
            # Set perturbation vector
            perturb[p] = e
            self.set_params(params_init + perturb)
            loss2 = self.square_error(X, y)
            
            self.set_params(params_init - perturb)
            loss1 = self.square_error(X, y)

            # Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2 * e)

            # Return the value we changed to zero:
            perturb[p] = 0
            
        # Return Params to original value:
        self.set_params(params_init)

        return numgrad 


    def get_params(self):
        # Concat all weights and biases and return them as 1-D array
        return np.concatenate([*[w.ravel() for w in self.Ws],
                               *[b         for b in self.bs]])
    
    def set_params(self, params):
        # Restores neural network parameters
        # from a given 1-D array (shape must match self.shape).
        offset = 0
        # Restore weights
        for i in range(1, len(self.shape)):
            wsize = self.shape[i - 1] * self.shape[i]
            self.Ws[i - 1] = params[offset:offset + wsize]\
                            .reshape(self.shape[i - 1], self.shape[i])                       
            offset += wsize
        # Restore biases
        for i in range(1, len(self.shape)):
            self.bs[i - 1] = params[offset:offset + self.shape[i]]
            offset += self.shape[i]


    def callback(self, params):
        self.set_params(params)
        self.J.append(self.square_error(self.X, self.y))
    
    def objective(self, params, X, y):
        self.set_params(params)
        #self._forward(X)
        cost = self.square_error(X, y)  
        grads = self.compute_grads(X, y)    
        return cost, grads
        
    def train(self, X, y, maxiter=200):
        self.X = X
        self.y = y
        self.J = []

        params0 = self.get_params()
        options = {'maxiter':maxiter, 'disp':True}

        self.opt_results = minimize(self.objective, params0, jac=True, method='BFGS',\
                                    args=(X, y), options=options, callback=self.callback)    

        self.set_params(self.opt_results.x)


    # Utils
    def compute_accuracy(self, X_test, y_test):
        self._forward(X_test)
        num_correct = 0
        for i, j in zip(y_test, self.y_hat):
            if i.argmax() == j.argmax():
                num_correct += 1
        return float(num_correct) / len(y_test)


    def plot_cost(self):
        if len(self.J) > 0:
            plt.plot(self.J)
            plt.title('Optimization Results')
            plt.xlabel('Iterations')
            plt.ylabel('Cost/Loss')
            plt.show()
        

# Debug
if __name__ == '__main__':
    from iris import IrisDF
    df = IrisDF()
    Xs = df.X_train#np.array([[0.68619022, 0.31670318, 0.61229281, 0.232249 ]])
    ys = df.y_train#np.array([[1, 0, 0]])
    nn = DNN(shape=[4, 6, 3], debug=False)
    print(nn.objective(nn.get_params(), Xs, ys)[1])
    print()
    print(nn.compute_num_grads(Xs, ys))