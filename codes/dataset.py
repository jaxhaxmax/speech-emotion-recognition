import os
import numpy as np

from features import extract_mfcc_fixed


# 🔹 CHANGE THIS PATH if your dataset is elsewhere
DATASET_DIR = r"C:\Users\tanma\OneDrive\Desktop\speech-emotion-recognition\Dataset"

# Emotion mapping for RAVDESS
EMOTION_MAP = {
    1: 0,  # neutral
    2: 1,  # calm
    3: 2,  # happy
    4: 3,  # sad
    5: 4,  # angry
    6: 5,  # fearful
    7: 6,  # disgust
    8: 7   # surprised
}


def load_dataset():
    X = []  # features
    y = []  # labels

    for root, dirs, files in os.walk(DATASET_DIR):
        for file in files:
            if file.endswith(".wav"):

                file_path = os.path.join(root, file)

                # 🔹 Extract MFCC features
                mfcc = extract_mfcc_fixed(file_path)
                X.append(mfcc)

                # 🔹 Extract emotion label from filename
                emotion_code = int(file.split("-")[2])
                label = EMOTION_MAP[emotion_code]
                y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y


if __name__ == "__main__":
    X, y = load_dataset()

    print("Dataset loaded")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # sanity check
    print("First label:", y)
from collections import Counter

print("Class distribution:")
print(Counter(y))
