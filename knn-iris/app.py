import pandas as pd
import math

def load_data(csv_path):
    """
    Load the data from a CSV file (with columns:
    [SepalLength, SepalWidth, PetalLength, PetalWidth, Name])
    and return a pandas DataFrame.
    """
    df = pd.read_csv(csv_path)
    return df

def train_test_split(df, test_size=0.2, random_seed=42):
    """
    Shuffle and split the DataFrame into train and test sets.
    test_size=0.2 means 20% of the data is used for testing.
    """
    df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    split_index = int(len(df_shuffled) * (1 - test_size))
    train_df = df_shuffled.iloc[:split_index].reset_index(drop=True)
    test_df  = df_shuffled.iloc[split_index:].reset_index(drop=True)
    return train_df, test_df

def euclidean_distance(x1, x2):
    """
    Compute the Euclidean distance between two points (x1, x2).
    Each should be a list/array of numeric features.
    """
    distance = 0.0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return math.sqrt(distance)

def get_neighbors(train_df, test_sample, k):
    """
    Return the k nearest neighbors to 'test_sample' 
    from the training DataFrame train_df.
    
    test_sample is typically 5 elements [SepalLength, SepalWidth, PetalLength, PetalWidth, Label/None].
    We only need the first 4 for distance calculation.
    """
    distances = []
    # Separate features and labels in the training DataFrame
    train_features = train_df.iloc[:, :-1].values  # all but last column
    train_labels   = train_df.iloc[:, -1].values   # last column = label

    # Extract just the features from the test sample (ignoring its label/None)
    test_features = test_sample[:-1]
    
    # Compute distance to every point in the training set
    for i in range(len(train_features)):
        dist = euclidean_distance(test_features, train_features[i])
        label = train_labels[i]
        distances.append((dist, label))
        
    # Sort by distance and pick the first k
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    return neighbors

def predict_classification(train_df, test_sample, k):
    """
    Predict the class (label) for a single test_sample
    by majority vote of the k nearest neighbors.
    
    test_sample should have 5 elements, with the last element as None or an unused label:
    [sepal_length, sepal_width, petal_length, petal_width, None]
    """
    neighbors = get_neighbors(train_df, test_sample, k)
    class_votes = {}
    for dist, label in neighbors:
        class_votes[label] = class_votes.get(label, 0) + 1
    predicted_label = max(class_votes, key=class_votes.get)
    return predicted_label

def evaluate_knn(train_df, test_df, k):
    """
    Evaluate the KNN model by predicting each row in test_df
    and calculating the classification accuracy.
    """
    correct = 0
    total = len(test_df)
    
    for i in range(total):
        test_sample = test_df.iloc[i].values  # features + label
        actual_label = test_sample[-1]
        predicted_label = predict_classification(train_df, test_sample, k)
        if predicted_label == actual_label:
            correct += 1
    
    accuracy = correct / total
    return accuracy

def summarize_classes(train_df):
    """
    Print out each unique class in the training data
    and the average value of each feature for that class.
    """
    label_col = train_df.columns[-1]
    feature_cols = train_df.columns[:-1]
    
    group_stats = train_df.groupby(label_col)[feature_cols].mean()
    
    print("Classes in the training data and their average feature values:")
    for class_label, row in group_stats.iterrows():
        print(f"  Class: {class_label}")
        for col in row.index:
            print(f"    {col} = {row[col]:.2f}")
        print()

def get_float_input(prompt):
    """
    Repeatedly prompts the user for a floating-point number 
    until they enter valid input.
    """
    while True:
        value_str = input(prompt)
        try:
            return float(value_str)
        except ValueError:
            print("Invalid input. Please enter a valid number (e.g. 5.1).")

def get_yes_no_input(prompt):
    """
    Repeatedly prompt for 'y' or 'n' and return a boolean.
    """
    while True:
        choice = input(prompt).strip().lower()
        if choice == 'y':
            return True
        elif choice == 'n':
            return False
        else:
            print("Invalid choice. Please enter 'y' or 'n'.")

def main():
    csv_path = "data/iris.csv"  # Adjust if needed

    # Load the data
    df = load_data(csv_path)

    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=0.2, random_seed=42)

    # Summarize classes in the training set
    summarize_classes(train_df)

    # KNN evaluation
    k = 5
    accuracy = evaluate_knn(train_df, test_df, k)
    print(f"KNN classification accuracy (k={k}): {accuracy:.2f}")

    # Continuously ask for new input or quit
    while True:
        if not get_yes_no_input("\nWould you like to classify a new sample? (y/n): "):
            print("Exiting. Goodbye!")
            break

        print("Enter measurements for a new sample:")
        sepal_length = get_float_input("  Sepal Length (cm): ")
        sepal_width  = get_float_input("  Sepal Width  (cm): ")
        petal_length = get_float_input("  Petal Length (cm): ")
        petal_width  = get_float_input("  Petal Width  (cm): ")

        # Prepare the sample with an empty label placeholder
        new_sample = [sepal_length, sepal_width, petal_length, petal_width, None]

        # Predict the class for this new sample
        predicted_label = predict_classification(train_df, new_sample, k)
        print(f"The predicted class for your input is: {predicted_label}")

if __name__ == "__main__":
    main()
