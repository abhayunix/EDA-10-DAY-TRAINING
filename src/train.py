"""Main training script for ML model."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

sys.path.insert(0, str(Path(__file__).parent))
from utils import split_data, scale_features


def main():
    """Train and evaluate a random forest classifier on Iris dataset."""
    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print("Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Save model
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "model.pkl")
    joblib.dump(scaler, models_dir / "scaler.pkl")
    print("\nModel saved to models/model.pkl")


if __name__ == "__main__":
    main()
