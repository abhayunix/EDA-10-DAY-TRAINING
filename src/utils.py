"""Utility functions for ML project."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """Load data from CSV file."""
    return pd.read_csv(filepath)


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train, X_test):
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
