import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# PATH Configuration
DATASET_PATH = os.path.join('telco_churn_preprocessing', 'telco_churn_preprocessed.csv')

# MLFlow Configuration
EXPERIMENT_NAME = 'telco_churn_experiment_baseline'
mlflow.set_tracking_uri('http://127.0.0.1:5001/')
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()

def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load Preprocessed dataset and split into features and target
    """
    df = pd.read_csv(path)
    features = df.drop(columns=["Churn"])
    target = df["Churn"]
    return features, target

def split_data(
    features: pd.DataFrame, 
    target: pd.Series,) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into training and testing sets with stratification
    """
    return train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

def train_model(
    x_train: pd.DataFrame, y_train:pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    return model

def evaluate_model(
    model: RandomForestClassifier, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """
    Evaluate trained model on test set.
    """
    predictions = model.predict(x_test)
    return{
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions)
    }

def main() -> None:
    """
    End-to-end model training pipeline with MLFlow tracking.
    """
    features, target = load_data(DATASET_PATH)
    x_train, x_test, y_train, y_test = split_data(features, target)

    with mlflow.start_run(run_name="baseline_random_forest"):
        model = train_model(x_train, y_train)
        metrics = evaluate_model(model, x_test, y_test)

        # Log additional metrics manually
        mlflow.log_metrics(metrics)

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")

if __name__ == '__main__':
    main()