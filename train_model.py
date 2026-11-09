# train_model.py  -  works with several data sources and prints clear errors
import io
import os
import sys
import zipfile
import urllib.request

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

UCI_ZIP = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"


# ---------------------------------------------------------------
# Activity A: Retrieve the dataset (3 fallbacks)
# ---------------------------------------------------------------
def find_math_csv_in_zip(data):
    """Look for student-mat.csv inside a zip (including nested zips)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            if name.lower().endswith("student-mat.csv"):
                return pd.read_csv(z.open(name), sep=";")
        for name in z.namelist():
            if name.lower().endswith(".zip"):
                result = find_math_csv_in_zip(z.read(name))
                if result is not None:
                    return result
    return None


def get_data():
    # 1) Local file (download student-mat.csv manually and put it here)
    if os.path.exists("student-mat.csv"):
        print("Using local student-mat.csv")
        return pd.read_csv("student-mat.csv", sep=";")

    # 2) Direct download from UCI (Mathematics file, 395 rows)
    try:
        print("Downloading from UCI ...")
        with urllib.request.urlopen(UCI_ZIP, timeout=60) as r:
            df = find_math_csv_in_zip(r.read())
        if df is not None:
            return df
        print("student-mat.csv not found inside the zip.")
    except Exception as e:
        print("Direct download failed:", e)

    # 3) ucimlrepo package
    try:
        print("Trying ucimlrepo ...")
        from ucimlrepo import fetch_ucirepo
        s = fetch_ucirepo(id=320)
        df = pd.concat([s.data.features, s.data.targets], axis=1)
        return df.loc[:, ~df.columns.duplicated()]
    except Exception as e:
        print("ucimlrepo failed:", e)

    sys.exit(
        "\nCould not get the dataset. Fix: download the zip from\n"
        "https://archive.ics.uci.edu/dataset/320/student+performance\n"
        "extract 'student-mat.csv' into this folder, then run again."
    )


df = get_data()

for col in ("G1", "G2", "G3"):
    if col not in df.columns:
        sys.exit(f"Column {col} missing. Columns found: {list(df.columns)}")

df.to_csv("student_data.csv", index=False)
print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())

# ---------------------------------------------------------------
# Activity B: Preprocess, train, evaluate
# ---------------------------------------------------------------
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)          # 1 = at risk
X = df.drop(columns=["G3"])              # never use G3 as predictor

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight="balanced")),
])
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"], zero_division=0))

joblib.dump(model, "student_model.pkl")
joblib.dump(list(X.columns), "feature_columns.pkl")
print("Saved student_model.pkl")
