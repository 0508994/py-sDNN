import numpy as np
from copy import deepcopy
from scipy.optimize import minimize

np.seterr(all='ignore')

def sigmoid(x):
    return np.exp(-x) / ((1 + np.exp(-x))**2)

def dsigmoid(x):
    return x * (1. - x)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1. - x * x

def relu(x):
    return x * (x > 0)

def drelu(x):
    return 1. * (x > 0)

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis]
    return e_x / div

class DNN:
    def __init__(self, shape, debug=False):
        assert len(shape) > 2
        self.shape = shape
        self.Ws = []
        self.bs = []

        for i in range(1, len(shape)):
            if debug:
                self.Ws.append(np.ones([shape[i - 1], shape[i]])) #* np.sqrt(2.0 / shape[i - 1]))
                self.bs.append(np.zeros(shape[i])) #* np.sqrt(2.0 / shape[i - 1]))
            else:
                # Xavier Glorot initialization
                # https://theneuralperspective.com/2016/11/11/weights-initialization/
                # https://www.youtube.com/watch?v=s2coXdufOzE
                self.Ws.append(np.random.randn(shape[i - 1], shape[i]) * np.sqrt(2.0 / shape[i - 1]))
                self.bs.append(np.random.randn(shape[i])) #* np.sqrt(2.0 / shape[i - 1]))

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def _forward(self, X):
        # # Propagate inputs through the network
        self.zs = []    # dot products between layers
        self.acts = [X] # activation function applied on dot product between layers
        tmp = X
        for i in range(len(self.Ws)):
            tmp = np.dot(tmp, self.Ws[i]) + self.bs[i]
            self.zs.append(tmp)
            if i != len(self.Ws) - 1 # TODO: find a better way to do this........
                tmp = relu(tmp)
                self.acts.append(tmp)
        self.y_hat = softmax(tmp)

    def compute_cost(self, X, y):
        #self._forward(X)
        return 0.5 * sum((y - self.y_hat)**2)

    # https://stackoverflow.com/questions/3775032/how-to-update-the-bias-in-neural-network-backpropagation
    def compute_grads(self, X, y):
        #self._forward(X)
        #self.acts.pop()

        delta = np.multiply(-(y - self.y_hat), drelu(self.zs[-1]))
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
            grads_w.insert(0, dJdW) # prepend to the grads list
            grads_b.insert(0, dJdb)    

        return np.concatenate([*grads_w, *grads_b], axis=None)


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
        '''
        Callback argument for each iteration of the scipy.optimize.minimize
        Called after each iteration, as callback(xk), ----> xk in this case (all my weights and biases)
        where xk is the current parameter vector.
        
        Note to self:
        set params and save cost in a self.J if you want
        '''
        pass
    
    def objective(self, params, X, y):
        self.set_params(params)
        self._forward(X)
        cost = self.compute_cost(X, y)      
        grads = self.compute_grads(X, y)    
        return cost, grads

        
    def train(self, X, y, maxiter=200):
        '''
            Nije bitno kakao pakujem tezine i biase u params0,
            zato sto ih sam vracam na mesto sa set_params
        '''
        self.X = X
        self.y = y

        #self.J = []

        params0 = self.get_params()
        options = {'maxiter':maxiter, 'disp':True }

        self.opt_results = minimize(self.objective, params0, jac=True, method='BFGS',\
                                    args=(X, y), options=options, callback=self.callback)    

        self.set_params(self.opt_results.x)
        

# Debug
if __name__ == '__main__':
    nn = DNN(shape=[2, 3, 1], debug=True)
    #nn._forward([[1.0, 1.0, 1.0, 1.0]])
    # nn1 = DNN(shape=[2, 2, 3])
    # nn1.set_params(nn.get_params())
    # print(nn.bs[1])
    # print(nn1.bs[1])
    print(nn.compute_grads(np.array([[2.0, 3.1]]), np.array([[2.0]])))
    # nn._forward(np.array([[2.0, 3.1]]))
    # print(nn.y_hat.shape)