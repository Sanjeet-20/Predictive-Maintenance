import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score

# =========================
# 1. Column names
# =========================
columns = (
    ["engine_id", "cycle"]
    + [f"setting{i}" for i in range(1, 4)]
    + [f"sensor{i}" for i in range(1, 22)]
)

# =========================
# 2. Load training data
# =========================
df = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=columns
)

# =========================
# 3. RUL + failure label
# =========================
max_cycle = df.groupby("engine_id")["cycle"].max()

df["RUL"] = df.apply(
    lambda r: max_cycle[r.engine_id] - r.cycle,
    axis=1
)

# Failure within next 20 cycles
df["failure"] = (df["RUL"] <= 20).astype(int)

# =========================
# 4. Use TOP 5 sensors
# =========================
sensor_cols = ["sensor14", "sensor9", "sensor4", "sensor7", "sensor12"]

# =========================
# 5. Engine-level split
# =========================
engines = df["engine_id"].unique()

train_eng, test_eng = train_test_split(
    engines, test_size=0.2, random_state=42
)

train_df = df[df.engine_id.isin(train_eng)]
test_df  = df[df.engine_id.isin(test_eng)]

X_train = train_df[sensor_cols]
y_train = train_df["failure"]

X_test = test_df[sensor_cols]
y_test = test_df["failure"]

# =========================
# 6. Train Random Forest
# =========================
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

# =========================
# 7. Calibrate probabilities
# =========================
model = CalibratedClassifierCV(
    estimator=rf,
    method="isotonic",
    cv=3
)

model.fit(X_train, y_train)

# =========================
# 8. Choose DATA-DRIVEN threshold
# =========================
y_prob = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# choose threshold where precision >= 0.7
idx = np.where(precision >= 0.7)[0]
THRESHOLD = thresholds[idx[0]] if len(idx) else 0.1

print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))
print("Chosen threshold:", round(THRESHOLD, 4))
print("Model classes:", model.classes_)

# =========================
# 9. Save model + threshold
# =========================
joblib.dump(model, "model.joblib")

with open("threshold.txt", "w") as f:
    f.write(str(THRESHOLD))

print("Model and threshold saved successfully")
