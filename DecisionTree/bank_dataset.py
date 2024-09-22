import csv
import numpy as np
from decision_tree import DecisionTree
from collections import Counter

majority_value = []

def load_data(file_path, handle_missing=False):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row[:-1])
            labels.append(row[-1])
    
    if handle_missing:
        fill_missing_values(data)  
    
    return data, labels


def fill_missing_values(data):
    global majority_value
    majority_value = [Counter([row[i] for row in data if row[i] != 'unknown']).most_common(1)[0][0] for i in range(len(data[0]))]
    
    for row in data:
        for i in range(len(row)):
            if row[i] == 'unknown':
                row[i] = majority_value[i]


def convert_to_binary(data, numeric_indices):
    for i in numeric_indices:
        column_values = [float(row[i]) for row in data if row[i] != 'unknown']
        median = np.median(column_values)
        for row in data:
            if row[i] != 'unknown':
                row[i] = '1' if float(row[i]) > median else '0'
            else:
                row[i] = '1' if majority_value[i] > median else '0'  
    return data


def evaluate_decision_tree_bank(max_depth, heuristic, handle_missing=False):
    train_data, train_labels = load_data('bank/train.csv', handle_missing)
    test_data, test_labels = load_data('bank/test.csv', handle_missing)
    
    numeric_indices = [0, 5, 9, 11, 12, 13, 14]
    if handle_missing:
        fill_missing_values(train_data)  
        
    train_data = convert_to_binary(train_data, numeric_indices)
    test_data = convert_to_binary(test_data, numeric_indices)
    
    dt = DecisionTree(max_depth=max_depth, heuristic=heuristic)
    dt.fit(train_data, train_labels)
    
    train_predictions = dt.predict(train_data)
    test_predictions = dt.predict(test_data)
    
    train_error = sum([1 for i in range(len(train_labels)) if train_labels[i] != train_predictions[i]]) / len(train_labels)
    test_error = sum([1 for i in range(len(test_labels)) if test_labels[i] != test_predictions[i]]) / len(test_labels)
    
    return train_error, test_error

if __name__ == "__main__":
    # Question 3a: Treat 'unknown' as a valid value
    print("Question 3a: Treating 'unknown' as valid value")
    print("Heuristic | Depth | Train Error | Test Error")
    print("-------------------------------------------")
    for heuristic in ['information_gain', 'gini_index', 'majority_error']:
        for depth in range(1, 17):
            train_err, test_err = evaluate_decision_tree_bank(depth, heuristic)
            print(f"{heuristic: <12} | {depth: <5} | {train_err: <12.4f} | {test_err: <10.4f}")
    
    # Question 3b: Treat 'unknown' as missing value
    print("\nQuestion 3b: Treating 'unknown' as missing value")
    print("Heuristic | Depth | Train Error | Test Error")
    print("-------------------------------------------")
    for heuristic in ['information_gain', 'gini_index', 'majority_error']:
        for depth in range(1, 17):
            train_err, test_err = evaluate_decision_tree_bank(depth, heuristic, handle_missing=True)
            print(f"{heuristic: <12} | {depth: <5} | {train_err: <12.4f} | {test_err: <10.4f}")
