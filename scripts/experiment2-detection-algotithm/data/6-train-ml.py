import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_enter_classifier(csv_path):
    """
    Trains a classifier to detect 'enter' labels from a dataset of features.

    Args:
        csv_path (str): The path to the CSV file containing features and labels.
    """
    # Load the dataset
    df = pd.read_csv(csv_path)

    # Separate features (X) and the target variable (y)
    # The 'enter_label' is the target. 'leave_label' is ignored.
    X = df.drop(columns=['enter_label', 'leave_label'])
    # y = df['leave_label']
    y = df['enter_label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the classifier (using a RandomForestClassifier as a robust choice)
    classifier = RandomForestClassifier(n_estimators=20, random_state=42, class_weight='balanced_subsample')

    # Train the classifier
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    print("Model Training and Evaluation Report:")
    print("-------------------------------------")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Example usage:
# Assuming 'feature_dataset.csv' is in the specified directory
train_enter_classifier("scripts/experiment2-detection-algotithm/data/feature_dataset.csv")