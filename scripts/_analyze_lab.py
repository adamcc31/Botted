"""Analyze experiment results and identify golden config."""

winners = [
    {"feat": "top5", "hyper": "balanced", "cal": "isotonic", "cr": 0.25, "folds": 3,
     "mean_auc": 0.7286, "std_auc": 0.0177, "ece": 0.0002, "oof_auc": 0.7168},
    {"feat": "top5", "hyper": "balanced", "cal": "isotonic", "cr": 0.30, "folds": 3,
     "mean_auc": 0.7370, "std_auc": 0.0388, "ece": 0.0001, "oof_auc": 0.7205},
    {"feat": "top5", "hyper": "balanced", "cal": "isotonic", "cr": 0.30, "folds": 5,
     "mean_auc": 0.7399, "std_auc": 0.0602, "ece": 0.0001, "oof_auc": 0.7446},
    {"feat": "top5", "hyper": "deep_reg", "cal": "isotonic", "cr": 0.25, "folds": 3,
     "mean_auc": 0.7001, "std_auc": 0.0166, "ece": 0.0000, "oof_auc": 0.6814},
    {"feat": "top5", "hyper": "deep_reg", "cal": "isotonic", "cr": 0.30, "folds": 5,
     "mean_auc": 0.7179, "std_auc": 0.0526, "ece": 0.0002, "oof_auc": 0.7189},
    {"feat": "top7", "hyper": "deep_reg", "cal": "isotonic", "cr": 0.25, "folds": 3,
     "mean_auc": 0.6996, "std_auc": 0.0186, "ece": 0.0002, "oof_auc": 0.6830},
    {"feat": "top7", "hyper": "deep_reg", "cal": "isotonic", "cr": 0.30, "folds": 5,
     "mean_auc": 0.7129, "std_auc": 0.0607, "ece": 0.0001, "oof_auc": 0.7179},
]

print("REAL WINNERS (OOF AUC > 0.65 AND pass both gates):")
print("=" * 90)
print(f"{'Config':<55} {'AUC':>6} {'std':>6} {'ECE':>8} {'OOF':>6}")
print("-" * 90)
for w in sorted(winners, key=lambda x: -x["oof_auc"]):
    tag = f"{w['feat']}/{w['hyper']}/{w['cal']}/cr={w['cr']}/f={w['folds']}"
    print(f"{tag:<55} {w['mean_auc']:>6.4f} {w['std_auc']:>6.4f} {w['ece']:>8.4f} {w['oof_auc']:>6.4f}")

print()
best = max(winners, key=lambda x: x["oof_auc"])
print(f"GOLDEN CONFIG: {best['feat']}/{best['hyper']}/{best['cal']}/cr={best['cr']}/f={best['folds']}")
print(f"  OOF AUC: {best['oof_auc']} | std: {best['std_auc']} | ECE: {best['ece']}")
