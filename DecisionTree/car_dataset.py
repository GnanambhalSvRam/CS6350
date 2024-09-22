import csv
from decision_tree import DecisionTree

def load_data(file_path):
    data = []
    labels = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row[:-1])  
            labels.append(row[-1])  
    return data, labels

def evaluate_decision_tree(max_depth, heuristic):
    train_data, train_labels = load_data('car/train.csv')
    test_data, test_labels = load_data('car/test.csv')
    
    dt = DecisionTree(max_depth=max_depth, heuristic=heuristic)
    dt.fit(train_data, train_labels)
    
    train_predictions = dt.predict(train_data)
    test_predictions = dt.predict(test_data)
    
    train_error = sum([1 for i in range(len(train_labels)) if train_labels[i] != train_predictions[i]]) / len(train_labels)
    test_error = sum([1 for i in range(len(test_labels)) if test_labels[i] != test_predictions[i]]) / len(test_labels)
    
    return train_error, test_error

if __name__ == "__main__":
    for heuristic in ['information_gain', 'gini_index', 'majority_error']:
        print(f"Heuristic: {heuristic}")
        for depth in range(1, 7):
            train_err, test_err = evaluate_decision_tree(depth, heuristic)
            print(f"Depth: {depth}, Train Error: {train_err}, Test Error: {test_err}")
