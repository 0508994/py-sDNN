# Debug
import sys
sys.path.append('../')
from nn.sdnn import DNN
from nn.iris import IrisDF
from numpy.linalg import norm


# http://ufldl.stanford.edu/wiki/index.php/Gradient_checking_and_advanced_optimization
if __name__ == '__main__':   
    df = IrisDF()
    Xs = df.X_train #np.array([[0.68619022, 0.31670318, 0.61229281, 0.232249 ]])
    ys = df.y_train #np.array([[1, 0, 0]])
    nn = DNN(shape=[4, 6, 3])
    calc_grad = nn.objective(nn.get_params(), Xs, ys)[1]
    num_grad = nn.compute_num_grads(Xs, ys)
    print(calc_grad, end='\n\n')
    print(num_grad, end='\n\n')
    print(norm(calc_grad - num_grad) / (norm(calc_grad + num_grad))) # e-9 or less ! 