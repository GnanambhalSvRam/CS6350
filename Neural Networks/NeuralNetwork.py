import numpy as np
from os import makedirs
import matplotlib.pyplot as plt

class SigmoidActivationFunction:
    def __call__(self, x: float) -> float:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        sig = self.__call__(x)
        return sig * (1 - sig)

class IdentityActivationFunction:
    def __call__(self, x: float) -> float:
        return x

    def derivative(self, x):
        return 1

class ConnectedLayer:
    def __init__(self, input_size, output_size, activation_func, weight_init_method, include_bias=True):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_func = SigmoidActivationFunction() if activation_func == 'sigmoid' else IdentityActivationFunction()
        self.weights = self.initialize_weights(weight_init_method, include_bias)

    def initialize_weights(self, method, include_bias):
        shape = (self.input_size + 1, self.output_size + 1) if include_bias else (self.input_size + 1, self.output_size)
        if method == 'zeroes':
            return np.zeros(shape, dtype=np.float64)
        elif method == 'random':
            return np.random.standard_normal(shape)
        else:
            raise ValueError("Unknown weight initialization method")

    def __str__(self) -> str:
        return str(self.weights)
    
    def compute_output(self, x):
        return self.activation_func(np.dot(x, self.weights))
    
    def backward_pass(self, zs, gradients):
        delta = np.dot(gradients[-1], self.weights.T) * self.activation_func.derivative(zs)
        return delta
    
    def update_weights(self, learning_rate, zs, gradients):
        gradient = np.dot(zs.T, gradients)
        self.weights -= learning_rate * gradient
        return gradient

class NeuralNetworkModel:
    def __init__(self, layers):
        self.layers = layers

    def forward_pass(self, x): 
        zs = [np.atleast_2d(np.append(1, x))]
        for layer in self.layers:
            out = layer.compute_output(zs[-1])
            zs.append(out)
        return float(zs[-1]), zs

    def backward_pass(self, zs, target, learning_rate=0.1, display=False):
        gradients = [zs[-1] - target]
        for i in range(len(zs) - 2, 0, -1):
            delta = self.layers[i].backward_pass(zs[i], gradients)
            gradients.append(delta)
        gradients.reverse()

        for i, layer in enumerate(self.layers):
            grad = layer.update_weights(learning_rate, zs[i], gradients[i])
            if display: print(f"Layer {i + 1} Gradient: \n{grad}")

#Question 2a:

print("\n\nAnswer 2a: ")
net = NeuralNetworkModel([
    ConnectedLayer(input_size=4, output_size=5, activation_func='sigmoid', weight_init_method='random'), # input
    ConnectedLayer(input_size=5, output_size=5, activation_func='sigmoid', weight_init_method='random'), # hidden
    ConnectedLayer(input_size=5, output_size=1, activation_func='identity', weight_init_method='random', include_bias=False) # output
])

x = np.array([3.8481, 10.1539, -3.8561, -4.2228])
ystar = 0
y, activations = net.forward_pass(x)
net.backward_pass(activations, ystar, display=True)


#Question 2b:
print("\n\nAnswer 2b: ")
try: 
    makedirs("./out/")
except FileExistsError: 
    pass

def mse_loss(pred, target):
    return 0.5 * (pred - target) ** 2

def load_data(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(",")
            terms_flt = list(map(np.float64, terms))
            x.append(terms_flt[:-1])
            y.append(terms_flt[-1])
    return np.array(x), np.array(y)

train_x, train_y = load_data('train.csv')
test_x, test_y = load_data('test.csv')

def train_and_test(num_epochs, net, train_x, train_y, test_x, test_y, lr_0=0.1, d=1):
    train_errors = []
    test_errors = []

    for e in range(num_epochs):
        train_losses = []
        for i in range(len(train_x)):
            y, activations = net.forward_pass(train_x[i])
            train_losses.append(mse_loss(y, train_y[i]))
            lr = lr_0 / (1 + (lr_0 / d) * e)
            net.backward_pass(activations, train_y[i], lr)

        train_error = np.mean(train_losses)
        train_errors.append(train_error)

        test_losses = [mse_loss(net.forward_pass(test_x[i])[0], test_y[i]) for i in range(len(test_x))]
        test_error = np.mean(test_losses)
        test_errors.append(test_error)

        print(f"Epoch {e+1} - Training Error: {train_error:>8f}, Testing Error: {test_error:>8f}")

    return train_errors, test_errors

def plot_errors(train_errors, test_errors, width):
    fig, ax = plt.subplots()
    ax.plot(train_errors, label='Training Error')
    ax.plot(test_errors, label='Testing Error')
    ax.set_title(f"Training and Testing Errors - Width {width}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    plt.savefig(f"./out/width_{width}.png")

widths = [5, 10, 25, 50, 100]
learning_rates = [0.1, 0.1, 0.05, 0.01, 0.003]
decay = [1, 1, 1, 1, 2]

for width, lr, d in zip(widths, learning_rates, decay):
    print(f"\nWidth = {width}:\n-------------------------------")
    net = NeuralNetworkModel([
        ConnectedLayer(input_size=4, output_size=width, activation_func='sigmoid', weight_init_method='random'),
        ConnectedLayer(input_size=width, output_size=width, activation_func='sigmoid', weight_init_method='random'),
        ConnectedLayer(input_size=width, output_size=1, activation_func='identity', weight_init_method='random', include_bias=False)
    ])

    training_errors, testing_errors = train_and_test(50, net, train_x, train_y, test_x, test_y, lr_0=lr, d=d)
    plot_errors(training_errors, testing_errors, width)

plt.show()


#Question 2c:
print("\n\nAnswer 2c: ")
try: 
    makedirs("./out/")
except FileExistsError: 
    pass

def mse_loss(pred, target):
    return 0.5 * (pred - target) ** 2

def load_data(filename):
    x, y = [], []
    with open(filename, 'r') as f:
        for line in f:
            terms = line.strip().split(",")
            terms_flt = list(map(np.float64, terms))
            x.append(terms_flt[:-1])
            y.append(terms_flt[-1])
    return np.array(x), np.array(y)

train_x, train_y = load_data('train.csv')
test_x, test_y = load_data('test.csv')

def train_and_test(num_epochs, net, train_x, train_y, test_x, test_y, lr_0=0.1, d=1):
    train_errors = []
    test_errors = []

    for e in range(num_epochs):
        train_losses = []
        for i in range(len(train_x)):
            y, activations = net.forward_pass(train_x[i])
            train_losses.append(mse_loss(y, train_y[i]))
            lr = lr_0 / (1 + (lr_0 / d) * e)
            net.backward_pass(activations, train_y[i], lr)

        train_error = np.mean(train_losses)
        train_errors.append(train_error)

        test_losses = [mse_loss(net.forward_pass(test_x[i])[0], test_y[i]) for i in range(len(test_x))]
        test_error = np.mean(test_losses)
        test_errors.append(test_error)

        print(f"Epoch {e+1} - Training Error: {train_error:>8f}, Testing Error: {test_error:>8f}")

    return train_errors, test_errors

def plot_errors(train_errors, test_errors, width):
    fig, ax = plt.subplots()
    ax.plot(train_errors, label='Training Error')
    ax.plot(test_errors, label='Testing Error')
    ax.set_title(f"Training and Testing Errors - Width {width}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    plt.savefig(f"./out/width_{width}.png")

widths = [5, 10, 25, 50, 100]
learning_rates = [0.1, 0.1, 0.05, 0.01, 0.003]
decay = [1, 1, 1, 1, 2]

for width, lr, d in zip(widths, learning_rates, decay):
    print(f"\nWidth = {width}:\n-------------------------------")
    net = NeuralNetworkModel([
        ConnectedLayer(input_size=4, output_size=width, activation_func='sigmoid', weight_init_method='zeroes'),
        ConnectedLayer(input_size=width, output_size=width, activation_func='sigmoid', weight_init_method='zeroes'),
        ConnectedLayer(input_size=width, output_size=1, activation_func='identity', weight_init_method='zeroes', include_bias=False)
    ])

    training_errors, testing_errors = train_and_test(50, net, train_x, train_y, test_x, test_y, lr_0=lr, d=d)
    plot_errors(training_errors, testing_errors, width)

plt.show()