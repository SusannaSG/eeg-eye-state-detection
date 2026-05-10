import pandas as pd
import numpy as np
import os
import joblib
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# ── Config ───────────────────────────────────────────────
DATASET_PATH = "EEG-Eye-State.csv"
MODELS_DIR   = "models"
RANDOM_STATE = 42

# ── Load Data ────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"Shape: {df.shape}")

X = df.drop("eyeDetection", axis=1)
y = df["eyeDetection"]
FEATURE_NAMES = list(X.columns)

# ── Split & Scale ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Models ────────────────────────────────────────────────
models = {
    'logistic_regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'decision_tree':       DecisionTreeClassifier(random_state=RANDOM_STATE),
    'random_forest':       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    'svm':                 SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE),
    'knn':                 KNeighborsClassifier(n_neighbors=5),
    'naive_bayes':         GaussianNB(),
}

display_names = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree':       'Decision Tree',
    'random_forest':       'Random Forest',
    'svm':                 'SVM',
    'knn':                 'KNN',
    'naive_bayes':         'Naive Bayes',
}

# ── Train & Save ──────────────────────────────────────────
os.makedirs(MODELS_DIR, exist_ok=True)
results = {}
best_model_name = None
best_accuracy   = 0.0

print("\nTraining models...")
print("-" * 45)

for key, model in models.items():
    name = display_names[key]
    print(f"Training: {name}...")
    model.fit(X_train_scaled, y_train)

    y_pred   = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, output_dict=True)
    cm       = confusion_matrix(y_test, y_pred).tolist()

    print(f"  Accuracy: {accuracy * 100:.2f}%")

    results[name] = {
        'accuracy':  round(accuracy * 100, 2),
        'precision': round(report['weighted avg']['precision'] * 100, 2),
        'recall':    round(report['weighted avg']['recall']    * 100, 2),
        'f1_score':  round(report['weighted avg']['f1-score']  * 100, 2),
        'confusion_matrix': cm,
    }

    joblib.dump(model, os.path.join(MODELS_DIR, f"{key}.pkl"))

    if accuracy > best_accuracy:
        best_accuracy   = accuracy
        best_model_name = name

# ── Save Scaler & Metadata ────────────────────────────────
joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

metadata = {
    'feature_names': FEATURE_NAMES,
    'best_model':    best_model_name,
    'best_accuracy': round(best_accuracy * 100, 2),
    'results':       results,
    'class_labels':  {'0': 'Eyes Closed', '1': 'Eyes Open'},
}

with open(os.path.join(MODELS_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# ── Summary ───────────────────────────────────────────────
print("\n" + "=" * 45)
print("RESULTS SUMMARY")
print("=" * 45)
for name, r in results.items():
    print(f"{name:<22} → {r['accuracy']}%")

print(f"\nBest Model : {best_model_name} ({round(best_accuracy*100,2)}%)")
print("All models saved in /models folder!")