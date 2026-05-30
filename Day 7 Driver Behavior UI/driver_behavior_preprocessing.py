from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main():
    root = Path(__file__).resolve().parent
    data_path = root / "driver_behavior.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    if "behavior_label" not in df.columns:
        raise ValueError("Expected column 'behavior_label' in dataset")

    X = df.drop(columns=["behavior_label"])
    y = df["behavior_label"]

    # Splits: 60% train, 20% val, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)

    X_train.to_csv(artifacts / "X_train.csv", index=False)
    X_val.to_csv(artifacts / "X_val.csv", index=False)
    X_test.to_csv(artifacts / "X_test.csv", index=False)

    # Fit label encoder on full label set and save it
    le = LabelEncoder()
    le.fit(y)
    joblib.dump(le, artifacts / "label_encoder.joblib")

    # Save numeric-encoded labels for training/eval
    y_train_enc = le.transform(y_train)
    y_val_enc = le.transform(y_val)
    y_test_enc = le.transform(y_test)

    pd.DataFrame(y_train_enc).to_csv(artifacts / "y_train.csv", index=False)
    pd.DataFrame(y_val_enc).to_csv(artifacts / "y_val.csv", index=False)
    pd.DataFrame(y_test_enc).to_csv(artifacts / "y_test.csv", index=False)

    print("Saved preprocessing artifacts to:", artifacts.resolve())


if __name__ == "__main__":
    main()
