import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):

    # Remove unnecessary columns
    df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

    # Encode categorical features
    le_gender = LabelEncoder()
    le_geo = LabelEncoder()

    df["Gender"] = le_gender.fit_transform(df["Gender"])
    df["Geography"] = le_geo.fit_transform(df["Geography"])

    # Features and Target
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test