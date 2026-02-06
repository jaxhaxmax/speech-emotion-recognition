import numpy as np
from sklearn.model_selection import train_test_split

from dataset import load_dataset


# Load full dataset
X, y = load_dataset()

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
