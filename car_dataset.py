from decision_tree import initiate_decision_tree, calculate_error_rate
import pandas as pd

if __name__ == '__main__':
    # Load the car dataset
    columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    train_data = pd.read_csv("./car/train.csv", names=columns, header=None)
    test_data = pd.read_csv("./car/test.csv", names=columns, header=None)
    
    # Labels
    labels = ["unacc", "acc", "good", "vgood"]
    
    # Criteria
    criteria = ["entropy", "gini", "me"]
    
    for criterion in criteria:
        for depth in range(1, 7):
            decision_tree = initiate_decision_tree(train_data, "class", labels, depth, criterion)
            test_error = calculate_error_rate(decision_tree, test_data, 'class')
            train_error = calculate_error_rate(decision_tree, train_data, 'class')
            
            # Printing in table format
            print(f"{criterion:<10}{depth:<10}{test_error:<15.5f}{train_error:<15.5f}")