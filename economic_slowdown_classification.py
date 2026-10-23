# Import libraries for data handling and classification
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Load the datasets and check their dimensions and class distribution
train = pd.read_csv("ex2_train.csv")
test = pd.read_csv("ex2_test.csv")
variables = pd.read_csv("ex2_variables.csv")

# Verify shapes to ensure all files loaded correctly
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Variables shape:", variables.shape)

# Inspect target variable y distribution (count and proportion of slowdowns)

print("nTarget variable (y) counts:")
print(train["y"].value_counts())
print(
    "nProportion of slowdown quarters (y=1):",
    round(train["y"].mean(), 4),
)

# Prepare predictors and target
feature_cols = [c for c in train.columns if c not in ["id", "y"]]
X, y = train[feature_cols], train["y"]

# Set up 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# Evaluate the baseline model
scores = []

for tr_idx, va_idx in kf.split(X, y):
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
    preds = np.full(len(y_va), y_tr.mean())
    scores.append(brier_score_loss(y_va, preds))

cv_brier_baseline = np.mean(scores)
print(f"Baseline CV Brier Score: {cv_brier_baseline:.4f}")

# Evaluate Logistic Regression using 5-fold CV
scores_logreg = []

for tr_idx, va_idx in kf.split(X, y):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)

    probs = model.predict_proba(X_va)[:, 1]
    scores_logreg.append(brier_score_loss(y_va, probs))

# Calculate and print the average Brier score across the five folds
cv_brier_logreg = np.mean(scores_logreg)
print(f"Logistic Regression CV Brier Score: {cv_brier_logreg:.4f}")

# Compare alternative models using 5-fold CV

models = {
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    "KNN (k=15)": KNeighborsClassifier(n_neighbors=15),
    "Naive Bayes": GaussianNB()
}

results = {}

for name, model in models.items():
    scores = []

    for tr_idx, va_idx in kf.split(X, y):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_va)[:, 1]
        scores.append(brier_score_loss(y_va, probs))

    results[name] = np.mean(scores)

# Display Brier score comparison
print(f"Baseline: {cv_brier_baseline:.4f}")
print(f"Logistic Regression: {cv_brier_logreg:.4f}")

for name, score in results.items():
    print(f"{name}: {score:.4f}")

# Fit Logistic Regression using all 100 training observations
final_model = LogisticRegression(max_iter=1000)
final_model.fit(X, y)

# Predict the probability of a slowdown (y = 1)
test_probs = final_model.predict_proba(test[feature_cols])[:, 1]

# Keep probabilities within the required range
test_probs = np.clip(test_probs, 0.001, 0.999)

# Create the submission with exactly two required columns
submission = pd.DataFrame({
    "id": test["id"],
    "prediction": test_probs
})

# Check the predictions and submission format
print(submission.head())
print("Shape:", submission.shape)
print("Missing values:", submission.isnull().sum().to_dict())
print("Probability range:", submission["prediction"].min(),
      "to", submission["prediction"].max())

# Save the CSV without an extra index column
submission.to_csv("ex2_Kumar_Shreyansh.csv", index=False)
print("Submission saved successfully!")
