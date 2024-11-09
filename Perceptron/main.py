
from os import makedirs
import csv
import numpy as np
import perceptron

np.random.seed(33)

dataset_loc = "bank-note/"

train_x = []
train_y = []
with open(dataset_loc + "train.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        train_x.append(terms_flt[:-1])
        train_y.append(-1 if terms_flt[-1] == 0 else 1)

train_x = np.array(train_x)
train_y = np.array(train_y)

test_x = []
test_y = []
with open(dataset_loc + "test.csv", "r") as f:
    for line in f:
        terms = line.strip().split(",")
        terms_flt = list(map(lambda x : float(x), terms))
        test_x.append(terms_flt[:-1])
        test_y.append(-1 if terms_flt[-1] == 0 else 1)

test_x = np.array(test_x)
test_y = np.array(test_y)

print("\n\nResults for Standard Perceptron")
p = perceptron.Perceptron(train_x, train_y, r=0.1)
print(f"Learned weights: {p.weights}")
print(f"Training accuracy: {np.mean(train_y == p.predict(train_x))}")
print(f"Testing accuracy: {np.mean(test_y == p.predict(test_x))}")

print("\n\nResults for Voted Perceptron")
vp = perceptron.VotedPerceptron(train_x, train_y, r=0.1)
#print(f"\nLearned Weights and Counts: {vp.votes}")

print(f"Training accuracy: {np.mean(train_y == vp.predict(train_x))}")
print(f"Testing accuracy: {np.mean(test_y == vp.predict(test_x))}")

print("\n\nAveraged Perceptron")
ap = perceptron.AveragedPerceptron(train_x, train_y, r=0.1)
print(f"Learned weights: {ap.weights}")
print(f"Training accuracy: {np.mean(train_y == ap.predict(train_x))}")
print(f"Testing accuracy: {np.mean(test_y == ap.predict(test_x))} \n\n")
