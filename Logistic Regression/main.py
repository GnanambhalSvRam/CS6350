import pandas as pd
import numpy as np

class CustomLogisticModel:
    def __init__(self):
        self.lr = 0.003
        self.d = 0.1
        self.epoch = 100
        self.gamma = 0.1
        self.v = 1
    
    def update_lr(self, lr):
        self.lr = lr
    
    def update_d(self, d):
        self.d = d
    
    def update_epoch(self, epoch):
        self.epoch = epoch
    
    def update_gamma(self, gamma):
        self.gamma = gamma
    
    def update_v(self, v):
        self.v = v
    
    def fit_MAP(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros([1, dim])
        idx = np.arange(num_sample)
        for t in range(self.epoch):
            np.random.shuffle(idx)
            x = x[idx,:]
            y = y[idx]
            for i in range(num_sample):
                x_i = x[i,:].reshape([1, -1])
                tmp = y[i] * np.sum(np.multiply(w, x_i))
                g = - num_sample * y[i] * x_i / (1 + np.exp(tmp)) + w / self.v
                lr = self.lr / (1 + self.lr / self.d * t)
                w = w - lr * g
        return w.reshape([-1,1])
    
    def fit_ML(self, x, y):
        num_sample = x.shape[0]
        dim = x.shape[1]
        w = np.zeros([1, dim])
        idx = np.arange(num_sample)
        for t in range(self.epoch):
            np.random.shuffle(idx)
            x = x[idx,:]
            y = y[idx]
            for i in range(num_sample):
                tmp = y[i] * np.sum(np.multiply(w, x[i,:]))
                g = - num_sample * y[i] * x[i,:] / (1 + np.exp(tmp))
                lr = self.lr / (1 + self.lr / self.d * t)
                w = w - lr * g
        return w.reshape([-1,1])

def calculate_model_error(x, y, w):
    pred = np.matmul(x, w)
    pred[pred > 0] = 1
    pred[pred <= 0] = -1
    error = np.sum(np.abs(pred - np.reshape(y, (-1, 1)))) / 2 / y.shape[0]
    return error

def load_and_process_data(file):
    data = pd.read_csv(file, header=None).values
    num_col = data.shape[1]
    x = np.hstack((data[:, :-1], np.ones((data.shape[0], 1))))  # Append a column of 1s for bias
    y = data[:, -1]
    y = 2 * y - 1  # Transform labels
    return x, y

train_x, train_y = load_and_process_data('train.csv')
test_x, test_y = load_and_process_data('test.csv')

model = CustomLogisticModel()

v_list = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

for method in ['MAP', 'ML']:
    print(f"--- Method: {method} ---")
    for v in v_list:
        model.update_v(v)
        print(f'Variance: {v}')

        if method == 'MAP':
            w = model.fit_MAP(train_x, train_y)
        else:
            w = model.fit_ML(train_x, train_y)

        train_error = calculate_model_error(train_x, train_y, w)
        test_error = calculate_model_error(test_x, test_y, w)

        print(f'{method} Training Error: {train_error}, Test Error: {test_error}')