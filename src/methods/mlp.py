import numpy as np

class MLP:
    def __init__(self, dimensions, activations):
        """
        :param dimensions: list of dimensions of the neural net. (input, hidden layer, ... ,hidden layer, output)
        :param activations: list of activation functions. Must contain N-1 activation function, where N = len(dimensions).

        Example of one hidden layer with
        - 2 inputs
        - 10 hidden nodes
        - 5 outputs
        layers -->    [0,        1,          2]
        ----------------------------------------
        dimensions =  (2,     10,          5)
        activations = (      Sigmoid,      Sigmoid)
        """

        self.n_layers = len(dimensions)
        self.w = {}
        self.b = {}
        self.activations = {}
        self.learning_rate = None
        self.v_w = {}
        self.v_b = {}
        for l in range(1, self.n_layers):
            self.w[l] = np.random.randn(dimensions[l], dimensions[l-1]) / np.sqrt(dimensions[l - 1])
            self.b[l] = np.zeros(dimensions[l])
            self.activations[l] = activations[l-1]
            self.v_w[l] = np.zeros_like(self.w[l])
            self.v_b[l] = np.zeros_like(self.b[l])

    def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        a = {}
        z = {0: x}
        for l in range(1, self.n_layers):
            a[l] = np.dot(z[l-1], self.w[l].T) + self.b[l]
            z[l] = self.activations[l].forward(a[l])
        return a, z


    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """

        _, z = self.feed_forward(x)
        return z[self.n_layers - 1]


    def back_prop(self, z, a, y_true, loss):
        """
        The input dicts keys represent the layers of the net.
        a = { 0: x,
              1: f(w1(x) + b1)
              2: f(w2(a2) + b2)
              }
        :param a: (dict) w^T@x + b
        :param z: (dict) f(a)
        :param y_true: (array) One hot encoded truth vector.
        :param loss: Loss class with a static .gradient(y_true, y_pred) method.
        :return:
        """

        y_pred = z[self.n_layers - 1]
        delta = loss.gradient(y_true, y_pred) * self.activations[self.n_layers - 1].gradient(a[self.n_layers - 1])
        
        dw = np.dot(delta.T, z[self.n_layers - 2])

        update_params = {
            self.n_layers - 1: (dw, delta)
        }

        for l in reversed(range(1, self.n_layers - 1)):
            delta = np.dot(delta, self.w[l+1]) * self.activations[l].gradient(a[l])
            dw = np.dot(delta.T, z[l - 1])
            update_params[l] = (dw, delta)
        
        for k, v in update_params.items():
            self.update_w_b(k, v[0], v[1])


    def update_w_b(self, index, dw, delta):
        """
        Update weights and biases.
        :param index: (int) Number of the layer
        :param dw: (array) Partial derivatives
        :param delta: (array) Delta error.
        """
        db = np.sum(delta, 0)
        if hasattr(self, 'momentum') and self.momentum > 0:
            self.v_w[index] = self.momentum * self.v_w[index] + self.learning_rate * dw
            self.v_b[index] = self.momentum * self.v_b[index] + self.learning_rate * db
            self.w[index] -= self.v_w[index]
            self.b[index] -= self.v_b[index]
        else:
            self.w[index] -= self.learning_rate * dw
            self.b[index] -= self.learning_rate * db

    def fit(self, x, y_true, loss, epochs, batch_size, learning_rate=1e-3, momentum=0.0):
        """
        :param x: (array) Containing parameters
        :param y_true: (array) Containing one hot encoded labels.
        :param loss: Loss class (MSE, CrossEntropy etc.)
        :param epochs: (int) Number of epochs.
        :param batch_size: (int)
        :param learning_rate: (flt)
        :param momentum: (flt)
        """

        if not x.shape[0] == y_true.shape[0]:
            raise ValueError("Length of x and y arrays don't match")
        self.learning_rate = learning_rate
        self.momentum = momentum

        for i in range(epochs):
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x_ = x[indices]
            y_ = y_true[indices]

            for j in range(x.shape[0] // batch_size):
                k = j * batch_size
                l = (j + 1) * batch_size
                a, z = self.feed_forward(x_[k:l])
                self.back_prop(z, a, y_[k:l], loss)
