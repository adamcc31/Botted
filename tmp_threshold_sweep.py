import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from model_training.config import TARGET_COL
from model_training.features import build_features
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression

# Load Data
train_df = pd.read_csv('dataset/processed/train_split.csv', low_memory=False)
test_df  = pd.read_csv('dataset/processed/test_split.csv', low_memory=False)
train_df = build_features(train_df)
test_df = build_features(test_df)
y_test = (test_df[TARGET_COL] == 1.0).astype(int).values

V4_FEATURES_NO_SOURCE = ["entry_odds", "depth_ratio", "contest_urgency", "tfm_value", "obi_vol_interaction", "zone_id_code", "distance_usd"]

def train_model(features, df_train):
    calib_size = int(len(df_train) * 0.3)
    tr_sub = df_train.iloc[:-calib_size].copy()
    ca_sub = df_train.iloc[-calib_size:].copy()
    X_tr = tr_sub[features].values.astype(np.float32)
    y_tr = (tr_sub[TARGET_COL] == 1.0).astype(int).values
    X_ca = ca_sub[features].values.astype(np.float32)
    y_ca = (ca_sub[TARGET_COL] == 1.0).astype(int).values
    model = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.025, random_state=42)
    model.fit(X_tr, y_tr)
    p_raw_ca = model.predict_proba(X_ca)[:, 1]
    platt = IsotonicRegression(out_of_bounds="clip")
    platt.fit(p_raw_ca, y_ca)
    return model, platt

v4_ns_model, v4_ns_platt = train_model(V4_FEATURES_NO_SOURCE, train_df)

X_test = test_df[V4_FEATURES_NO_SOURCE].values.astype(np.float32)
p_raw = v4_ns_model.predict_proba(X_test)[:, 1]
p_cal = v4_ns_platt.predict(p_raw)
p_cal = np.clip(p_cal, 0.001, 0.999)

precision, recall, thresholds = precision_recall_curve(y_test, p_cal)

print("\n=== Precision/Recall Tradeoff (V4 No Source) ===")
targets = [0.60, 0.65, 0.68, 0.70, 0.75]
for target in targets:
    found = False
    for p, r, t in zip(precision, recall, thresholds):
        if p >= target:
            print(f"Target Prec >= {target:.2f}: threshold={t:.3f} → precision={p:.3f} recall={r:.3f}")
            found = True
            break
    if not found:
        print(f"Target Prec >= {target:.2f}: NOT FOUND")
