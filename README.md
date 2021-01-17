# py-sDNN

This repository represents an atempt to implement a simple, configurable, deep neural network from scratch, using nothing but pure python code along with some [SciPy](https://www.scipy.org/) tools. Primary resource used is Welch Labs [neural networks demystified series](https://www.youtube.com/watch?v=bxe2T-V8XRs&list=PLiaHhY2iBX9hdHaRr6b7XevZtgZRa1PoU) on YouTube. This is by no means an efficient implementation, and its only purpose is learning.



## Example usage
This example demonstrates usage on a [iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set).  IrisDF is a helper class that loads and preprocesses the data using [scikit-learn](https://scikit-learn.org/stable/) API.

```py
from nn.sdnn import DNN
from nn.iris import IrisDF

df = IrisDF()                   # load dataset
N = DNN(shape=[4, 10, 6, 3])    # create a neural network with input layer size of 4, two hidden layers, and 2 output classes (one-hot encoded)

N.train(df.X_train, df.y_train) # train the network
```

Behaviour of the cost/loss function over the number of iterations in the oprimization process for the given example is shown on the graph bellow.

![alt text](https://raw.githubusercontent.com/0508994/py-sDNN/master/tests/iris_loss.png)

More usage examples are avaiable in the test directory of this repo, in the form of jupyter notebooks.



