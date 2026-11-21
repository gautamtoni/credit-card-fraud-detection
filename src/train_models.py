
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    average_precision_score
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# 1. Load dataset
df = pd.read_csv("creditcard.csv")

# Remove rows with missing target
df = df.dropna(subset=["Class"]).copy()

# 2. Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"].astype(int)

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Calculate imbalance ratio from training data
scale_pos_weight = (
    (y_train == 0).sum() / (y_train == 1).sum()
)

# 4. Logistic Regression preprocessing
numeric_scaler = ColumnTransformer(
    transformers=[
        ("scale", StandardScaler(), ["Amount", "Time"])
    ],
    remainder="passthrough"
)

# 5. Define five models
models = {
    "Logistic Regression": make_pipeline(
        numeric_scaler,
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        )
    ),

    "Decision Tree": DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),

    "LightGBM": LGBMClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        verbosity=-1
    )
}

# 6. Train and evaluate
results = []

for name, model in models.items():

    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "PR-AUC": average_precision_score(
            y_test, probabilities
        )
    })

# 7. Display comparison
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="PR-AUC",
    ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# 8. Save results
results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print("\nResults saved to model_comparison.csv")
