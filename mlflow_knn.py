import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sys
import os

# Set the experiment name
mlflow.set_experiment("diabetes_experiment")

# Start an MLflow run
with mlflow.start_run() as run:
    # Read the dataset
    dataset_path = 'data/diabetes_data.csv'
    df = pd.read_csv(dataset_path)

    # Check if the file exists and log it as an artifact
    if os.path.exists('data/diabetes_data.csv'):
        print("Dataset file found. Logging artifact...")
        mlflow.log_artifact(dataset_path)
        mlflow.set_tag("Dataset", dataset_path)
    else:
        print("Dataset file not found. Skipping artifact logging.")

    # Print data shape
    print(f'Data shape: {df.shape}')

    # Split data into features and target
    X = df.drop(columns=['diabetes'])
    y = df['diabetes'].values

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    # Set KNN parameter
    n_neighbors = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    mlflow.log_param('n_neighbors', n_neighbors)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(knn, "knn_model")

    # Evaluate the model
    accuracy = knn.score(X_test, y_test)
    mlflow.log_metric('accuracy', accuracy)

    print(f'Model accuracy: {accuracy}')

    # Debug: Print run details
    print(f'Run ID: {run.info.run_id}')
    print(f'Artifact URI: {mlflow.get_artifact_uri()}')
    print(f'Artifacts logged to: {mlflow.get_artifact_uri("data/diabetes_data.csv")}')

print("Script execution completed.")
