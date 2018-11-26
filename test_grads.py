# Debug
# import sys
# sys.path.append('nn/')
from nn.sdnn import DNN
from nn.iris import IrisDF

if __name__ == '__main__':   
    df = IrisDF()
    Xs = df.X_train #np.array([[0.68619022, 0.31670318, 0.61229281, 0.232249 ]])
    ys = df.y_train #np.array([[1, 0, 0]])
    nn = DNN(shape=[4, 6, 3], debug=False)
    print(nn.objective(nn.get_params(), Xs, ys)[1])
    print()
    print(nn.compute_num_grads(Xs, ys))