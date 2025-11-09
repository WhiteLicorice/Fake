# model_metrics.py
import pickle
from pathlib import Path
import os
import numpy as np

MODEL_PATH = Path("models") / "LogisticRegression.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH.resolve()}")

# model size (MB)
model_size = MODEL_PATH.stat().st_size / (1024 ** 2)
print(f"Model size: {model_size:.2f} MB")

# load model correctly (open in binary mode)
with MODEL_PATH.open("rb") as f:
    model = pickle.load(f)

print(f"Loaded object type: {type(model)}")

if hasattr(model, "named_steps"):
    print("Pipeline steps:")
    for name, step in model.named_steps.items():
        print(f"  - {name}: {type(step)}")

# count parameters defensively
clf = model.named_steps['classifier']
n_params = 0
if hasattr(clf, "coef_"):
    try:
        coefficients = int(np.asarray(clf.coef_).size)
        n_params += coefficients
        print(f"coefficients: {coefficients}")
    except Exception:
        print("Warning: couldn't read model.coef_ shape.")
        
if hasattr(clf, "intercept_"):
    try:
        intercepts = int(np.asarray(clf.intercept_).size)
        n_params += intercepts
        print(f"intercepts: {intercepts}")
    except Exception:
        print("Warning: couldn't read model.intercept_ shape.")

print(f"Parameters: {n_params}")
