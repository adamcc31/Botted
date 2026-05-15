import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from model_training.config import TARGET_COL
from model_training.features import build_features
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Load Data
train_df = pd.read_csv('dataset/processed/train_split.csv', low_memory=False)
test_df  = pd.read_csv('dataset/processed/test_split.csv', low_memory=False)

train_df = build_features(train_df)
test_df = build_features(test_df)

y_test = (test_df[TARGET_COL] == 1.0).astype(int).values

# 2. Features Definitions
V4_FEATURES_WITH_SOURCE = [
    "entry_odds", "depth_ratio", "contest_urgency", "tfm_value", 
    "obi_vol_interaction", "zone_id_code", "distance_usd", "data_source_code"
]

V4_FEATURES_NO_SOURCE = [f for f in V4_FEATURES_WITH_SOURCE if f != 'data_source_code']

# 3. Models Setup
# V1 Prod
prod_dir = Path("models/alpha_v1")
prod_bundle = joblib.load(prod_dir / "model.pkl")
prod_model = prod_bundle["base_model"]
prod_platt = prod_bundle["platt"]
prod_features = prod_bundle["feature_names"]

# Helper to train and calibrate a model
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

# V4 with source
v4_ws_model, v4_ws_platt = train_model(V4_FEATURES_WITH_SOURCE, train_df)

# V4 no source
v4_ns_model, v4_ns_platt = train_model(V4_FEATURES_NO_SOURCE, train_df)

# 4. Evaluation Function
def get_metrics(model, platt, features, df):
    X = df[features].values.astype(np.float32)
    y = (df[TARGET_COL] == 1.0).astype(int).values
    
    p_raw = model.predict_proba(X)[:, 1]
    p_cal = platt.predict(p_raw)
    p_cal = np.clip(p_cal, 0.001, 0.999)
    
    y_pred = (p_cal >= 0.5).astype(int)
    
    return y_pred, p_cal

# 5. LANGKAH 1 & 2 & 3
results = []
models = [
    ('V1_Prod', prod_model, prod_platt, prod_features),
    ('V4_With_Source', v4_ws_model, v4_ws_platt, V4_FEATURES_WITH_SOURCE),
    ('V4_No_Source', v4_ns_model, v4_ns_platt, V4_FEATURES_NO_SOURCE)
]

for name, model, platt, features in models:
    y_pred, y_prob = get_metrics(model, platt, features, test_df)
    
    print(f"\n=== {name} Audit ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['LOSS', 'WIN']))
    
    win_pct = y_pred.mean() * 100
    print(f"{name}: predicts WIN {win_pct:.1f}% of time")
    
    prec_win = precision_score(y_test, y_pred)
    rec_win = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Precision (WIN)': prec_win,
        'Recall (WIN)': rec_win,
        'F1': f1,
        '% predicted WIN': win_pct
    })

# Naive baseline
naive_win_pct = (y_test == 1).mean() * 100
print(f"\nNaive WIN-always: {naive_win_pct:.1f}%")

# 6. LANGKAH 4 — Threshold calibration (V4 No Source)
print("\n=== Threshold Calibration (V4 No Source) ===")
y_pred_ns, y_prob_ns = get_metrics(v4_ns_model, v4_ns_platt, V4_FEATURES_NO_SOURCE, test_df)
precision, recall, thresholds = precision_recall_curve(y_test, y_prob_ns)

found = False
for p, r, t in zip(precision, recall, thresholds):
    if p >= 0.60:
        print(f"Optimal Threshold (Prec >= 0.60): threshold={t:.3f} → precision={p:.3f} recall={r:.3f}")
        found = True
        break
if not found:
    print("No threshold found with precision >= 0.60")

# 7. Final Table Output
print("\n=== FINAL AUDIT REPORT ===")
report_df = pd.DataFrame(results)
print(report_df.round(3).to_string(index=False))
