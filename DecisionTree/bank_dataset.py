import pandas as pd
import numpy as np
from decision_tree import initiate_decision_tree, calculate_error_rate, split_criteria, convert_num_feature

def handle_unknowns(data_set):
    """Replace 'unknown' values with the mode of the column."""
    for column in data_set.columns:
        if data_set[column].dtype == object:  # Only process categorical columns
            mode_value = data_set[column].mode()[0]  # Find the most frequent value
            data_set[column].replace('unknown', mode_value, inplace=True)
    return data_set

if __name__ == '__main__':
    # Define the columns and load the data
    bank_columns = ["age", "job", "marital", "education", "default", "balance", 
                    "housing", "loan", "contact", "day", "month", "duration", 
                    "campaign", "pdays", "previous", "poutcome", "class"]
    
    bank_num_columns = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
    label_bank = ["yes", "no"]
    
    train_data_bank = pd.read_csv("./bank/train.csv", names=bank_columns, header=None)
    test_data_bank = pd.read_csv("./bank/test.csv", names=bank_columns, header=None)
    
    # Part 1: Handle 'unknown' as a specific attribute value
    train_data_bank_known = convert_num_feature(train_data_bank.copy(), bank_num_columns)
    test_data_bank_known = convert_num_feature(test_data_bank.copy(), bank_num_columns)
    
    print("Results with 'unknown' as a specific value:")
    print("Method        Depth    Test Error      Train Error")
    for method in split_criteria:
        for depth in range(1, 17):
            decision_tree = initiate_decision_tree(train_data_bank_known, "class", label_bank, depth, method)
            test_error = calculate_error_rate(decision_tree, test_data_bank_known, 'class')
            train_error = calculate_error_rate(decision_tree, train_data_bank_known, 'class')
            print(f"{method:<10}{depth:<10}{test_error:<15.5f}{train_error:<15.5f}")
    
    # Part 2: Handle 'unknown' as missing value
    train_data_bank_unknown = handle_unknowns(train_data_bank.copy())
    test_data_bank_unknown = handle_unknowns(test_data_bank.copy())

    train_data_bank_unknown = convert_num_feature(train_data_bank_unknown, bank_num_columns)
    test_data_bank_unknown = convert_num_feature(test_data_bank_unknown, bank_num_columns)

    print("\nResults with 'unknown' as a missing value:")
    print("Method        Depth    Test Error      Train Error")
    for method in split_criteria:
        for depth in range(1, 17):
            decision_tree = initiate_decision_tree(train_data_bank_unknown, "class", label_bank, depth, method)
            test_error = calculate_error_rate(decision_tree, test_data_bank_unknown, 'class')
            train_error = calculate_error_rate(decision_tree, train_data_bank_unknown, 'class')
            print(f"{method:<10}{depth:<10}{test_error:<15.5f}{train_error:<15.5f}")
