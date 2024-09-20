import numpy as np
import pandas as pd

# Methods for splitting criteria
split_criteria = ["entropy", "gini", "me"]

# Function to compute subsets for an attribute and its values
def compute_subsets(data, target_column, class_labels, feature, max_depth):
    tree = {}
    value_counts = data[feature].value_counts()
    
    for value, count in value_counts.items():
        sub_data = data[data[feature] == value]
        
        max_label_count = 0
        max_label = None
        is_pure = False
        
        for label in class_labels:
            label_count = sub_data[sub_data[target_column] == label].shape[0]
            if label_count > max_label_count:
                max_label_count = label_count
                max_label = label
            if label_count == count:
                is_pure = True
                tree[value] = label
                data = data[data[feature] != value]
        
        if not is_pure:
            if max_depth > 1:
                tree[value] = "multi"
            else:
                tree[value] = max_label
    
    return data, tree

# Recursive function to build the decision tree
def build_tree(data, target_column, class_labels, prev_feature, tree_node, max_depth, criterion):
    if data.shape[0] > 0 and max_depth > 0:
        optimal_feature = select_optimal_feature(data, target_column, class_labels, criterion)
        data, subtree = compute_subsets(data, target_column, class_labels, optimal_feature, max_depth)
        
        if prev_feature is None:
            tree_node[optimal_feature] = subtree
            next_node = tree_node[optimal_feature]
        else:
            tree_node[prev_feature] = {}
            tree_node[prev_feature][optimal_feature] = subtree
            next_node = tree_node[prev_feature][optimal_feature]
        
        for key, value in next_node.items():
            if value == "multi":
                subset_data = data[data[optimal_feature] == key]
                build_tree(subset_data, target_column, class_labels, key, next_node, max_depth-1, criterion)

# Function to start the decision tree creation process
def initiate_decision_tree(data, target_column, class_labels, max_depth, criterion):
    tree = {}
    build_tree(data, target_column, class_labels, None, tree, max_depth, criterion)
    return tree

# Functions for calculating different criteria (entropy, gini, majority error)
def calculate_gini(data, target_column, class_labels):
    gini_value = 0
    total_instances = data.shape[0]
    
    for label in class_labels:
        label_count = data[data[target_column] == label].shape[0]
        gini_value += (label_count / total_instances) ** 2
    
    return 1 - gini_value

def calculate_entropy(data, target_column, class_labels):
    entropy = 0
    total_instances = data.shape[0]
    
    for label in class_labels:
        label_count = data[data[target_column] == label].shape[0]
        if label_count != 0:
            entropy -= (label_count / total_instances) * np.log2(label_count / total_instances)
    
    return entropy

def calculate_majority_error(data, target_column, class_labels):
    majority_error = 0
    total_instances = data.shape[0]
    
    for label in class_labels:
        label_count = data[data[target_column] == label].shape[0]
        majority_error = max(majority_error, label_count / total_instances)
    
    return 1 - majority_error

# Calculate information gain
def calculate_information_gain(data, target_column, class_labels, feature, criterion):
    unique_values = data[feature].unique()
    total_instances = data.shape[0]
    initial_calculation = 0
    
    if criterion == "entropy":
        initial_calculation = calculate_entropy(data, target_column, class_labels)
    elif criterion == "gini":
        initial_calculation = calculate_gini(data, target_column, class_labels)
    elif criterion == "me":
        initial_calculation = calculate_majority_error(data, target_column, class_labels)
    
    weighted_avg = 0
    for value in unique_values:
        subset = data[data[feature] == value]
        subset_size = subset.shape[0]
        subset_calculation = 0
        
        if criterion == "entropy":
            subset_calculation = calculate_entropy(subset, target_column, class_labels)
        elif criterion == "gini":
            subset_calculation = calculate_gini(subset, target_column, class_labels)
        elif criterion == "me":
            subset_calculation = calculate_majority_error(subset, target_column, class_labels)
        
        weighted_avg += (subset_size / total_instances) * subset_calculation
    
    return initial_calculation - weighted_avg

# Select the best feature based on information gain
def select_optimal_feature(data, target_column, class_labels, criterion):
    features = data.columns.drop(target_column)
    max_gain = -1
    optimal_feature = None
    
    for feature in features:
        info_gain = calculate_information_gain(data, target_column, class_labels, feature, criterion)
        if info_gain > max_gain:
            max_gain = info_gain
            optimal_feature = feature
    
    return optimal_feature

# Function for making predictions based on a decision tree
def make_prediction(tree, instance):
    if not isinstance(tree, dict):
        return tree
    else:
        root = next(iter(tree))
        feature_value = instance[root]
        if feature_value in tree[root]:
            return make_prediction(tree[root][feature_value], instance)
        else:
            return None

# Calculate error rate
def calculate_error_rate(tree, dataset, target_column):
    correct = 0
    incorrect = 0
    
    for idx, row in dataset.iterrows():
        prediction = make_prediction(tree, dataset.iloc[idx])
        if prediction == dataset[target_column].iloc[idx]:
            correct += 1
        else:
            incorrect += 1
    
    return incorrect / (correct + incorrect)

def convert_num_feature(data_set, num_feature_list):
    for num_feature in num_feature_list:
        cur_num_column = data_set.get(num_feature)
        temp_column = cur_num_column.copy(deep=True)
        mid_val = np.median(data_set.get(num_feature).to_numpy())
        temp_column.where(cur_num_column <= mid_val, "neg", inplace=True)
        temp_column.where(cur_num_column > mid_val, "pos", inplace=True)
        data_set[num_feature] = temp_column
    return data_set

def convert_unk_feature(data_set, feature_list):
    for feature in feature_list:
        # Check if 'unknown' exists in the feature's values
        if 'unknown' in data_set[feature].unique():
            # Get the current column
            feature_values = data_set[feature]
            
            # Identify the most frequent value excluding 'unknown'
            non_unknown_values = feature_values[feature_values != 'unknown']
            most_frequent_value = non_unknown_values.mode()[0]
            
            # Replace 'unknown' values with the most frequent value
            data_set[feature] = feature_values.replace('unknown', most_frequent_value)
    
    return data_set