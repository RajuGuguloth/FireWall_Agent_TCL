import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ==================================================
# CONFIG
# ==================================================
# Uses the pre-scaled enhanced .npy splits (25 features, 6 classes)
# Same data that train_cnn_gru_v3.py uses.
SPLITS_DIR  = "data/splits/enhanced"
MODELS_DIR  = Path("models/serialized")
RANDOM_SEED = 42
CLASSES     = ['BENIGN', 'BRUTE_FORCE', 'DDOS_HTTP_FLOOD',
               'DNS_TUNNELING', 'PORT_SCAN', 'SLOW_HTTP']

print("=" * 55)
print("TIER-2 ENSEMBLE TRAINING")
print("XGBoost + LightGBM + CatBoost  (majority vote)")
print("=" * 55)

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ==================================================
# LOAD DATA  (pre-scaled .npy from create_enhanced_splits.py)
# ==================================================
print("\nLoading data...")

X_train = np.load(f"{SPLITS_DIR}/X_train.npy")
X_val   = np.load(f"{SPLITS_DIR}/X_val.npy")
X_test  = np.load(f"{SPLITS_DIR}/X_test.npy")
y_train = np.load(f"{SPLITS_DIR}/y_train.npy")
y_val   = np.load(f"{SPLITS_DIR}/y_val.npy")
y_test  = np.load(f"{SPLITS_DIR}/y_test.npy")

# Load the encoder that was created by create_enhanced_splits.py
encoder = joblib.load(str(MODELS_DIR / "enhanced_label_encoder.pkl"))
CLASSES = list(encoder.classes_)  # use actual order from encoder

print(f"Train : {len(X_train)}")
print(f"Val   : {len(X_val)}")
print(f"Test  : {len(X_test)}")
print(f"Features : {X_train.shape[1]}")
print(f"Classes  : {CLASSES}")

num_classes = len(CLASSES)

# ==================================================
# XGBOOST
# ==================================================
print("\n" + "=" * 55)
print("Training XGBoost...")
print("=" * 55)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
dtest  = xgb.DMatrix(X_test,  label=y_test)

xgb_params = {
    "objective":        "multi:softmax",
    "num_class":        num_classes,
    "eval_metric":      "mlogloss",
    "max_depth":        6,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "seed":             RANDOM_SEED,
    "tree_method":      "hist",
}

xgb_model = xgb.train(
    xgb_params,
    dtrain,
    num_boost_round=300,
    evals=[(dval, "val")],
    early_stopping_rounds=20,
    verbose_eval=50,
)

xgb_pred = xgb_model.predict(dtest).astype(int).flatten()
print(f"XGBoost Test Accuracy: {accuracy_score(y_test, xgb_pred):.4f}")

# ==================================================
# LIGHTGBM
# ==================================================
print("\n" + "=" * 55)
print("Training LightGBM...")
print("=" * 55)

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val   = lgb.Dataset(X_val,   label=y_val, reference=lgb_train)

lgb_params = {
    "objective":        "multiclass",
    "num_class":        num_classes,
    "metric":           "multi_logloss",
    "num_leaves":       63,
    "learning_rate":    0.1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "verbose":          -1,
    "seed":             RANDOM_SEED,
}

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=300,
    valid_sets=[lgb_val],
    callbacks=[
        lgb.early_stopping(stopping_rounds=20, verbose=False),
        lgb.log_evaluation(period=50),
    ],
)

lgb_proba = lgb_model.predict(X_test)           # shape (n, num_classes)
lgb_pred  = np.argmax(lgb_proba, axis=1).flatten()
print(f"LightGBM Test Accuracy: {accuracy_score(y_test, lgb_pred):.4f}")

# ==================================================
# CATBOOST
# ==================================================
print("\n" + "=" * 55)
print("Training CatBoost...")
print("=" * 55)

cat_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.1,
    depth=6,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    random_seed=RANDOM_SEED,
    early_stopping_rounds=20,
    verbose=50,
)

cat_model.fit(
    X_train, y_train,
    eval_set=(X_val, y_val),
    use_best_model=True,
)

cat_pred = cat_model.predict(X_test).flatten().astype(int)
print(f"CatBoost Test Accuracy: {accuracy_score(y_test, cat_pred):.4f}")

# ==================================================
# ENSEMBLE (Majority Vote)
# ==================================================
print("\n" + "=" * 55)
print("Ensemble Prediction (Majority Vote)")
print("=" * 55)

xgb_pred = np.array(xgb_pred).flatten()
lgb_pred = np.array(lgb_pred).flatten()
cat_pred = np.array(cat_pred).flatten()

stacked       = np.stack([xgb_pred, lgb_pred, cat_pred], axis=0)   # (3, n)
ensemble_pred = stats.mode(stacked, axis=0, keepdims=False)[0].flatten()

ensemble_acc = accuracy_score(y_test, ensemble_pred)
print(f"Ensemble Test Accuracy: {ensemble_acc:.4f}")

# ==================================================
# DETAILED REPORT
# ==================================================
print("\n" + "=" * 55)
print("Detailed Classification Reports")
print("=" * 55)

print("\n--- XGBoost ---")
print(classification_report(y_test, xgb_pred, target_names=CLASSES, zero_division=0))

print("\n--- LightGBM ---")
print(classification_report(y_test, lgb_pred, target_names=CLASSES, zero_division=0))

print("\n--- CatBoost ---")
print(classification_report(y_test, cat_pred, target_names=CLASSES, zero_division=0))

print("\n--- Ensemble (Majority Vote) ---")
print(classification_report(y_test, ensemble_pred, target_names=CLASSES, zero_division=0))

# ==================================================
# SAVE ALL THREE MODELS
# ==================================================
print("\n" + "=" * 55)
print("Saving Models...")
print("=" * 55)

# XGBoost — save as JSON (native format)
xgb_path = MODELS_DIR / "tier2_xgboost.json"
xgb_model.save_model(str(xgb_path))
print(f"✅ XGBoost saved : {xgb_path}")

# LightGBM — save as .txt (native text format)
lgb_path = MODELS_DIR / "tier2_lightgbm.txt"
lgb_model.save_model(str(lgb_path))
print(f"✅ LightGBM saved: {lgb_path}")

# CatBoost — save as .cbm (native format)
cat_path = MODELS_DIR / "tier2_catboost.cbm"
cat_model.save_model(str(cat_path))
print(f"✅ CatBoost saved : {cat_path}")

# Also save ensemble metadata (class list + scores) for reference
import json
ensemble_meta = {
    "classes": CLASSES,
    "num_features": int(X_train.shape[1]),
    "ensemble_test_accuracy": float(ensemble_acc),
    "xgb_test_accuracy": float(accuracy_score(y_test, xgb_pred)),
    "lgb_test_accuracy": float(accuracy_score(y_test, lgb_pred)),
    "cat_test_accuracy": float(accuracy_score(y_test, cat_pred)),
}
meta_path = MODELS_DIR / "ensemble_meta.json"
with open(meta_path, "w") as f:
    json.dump(ensemble_meta, f, indent=2)
print(f"✅ Metadata saved : {meta_path}")

print("\n" + "=" * 55)
print("TIER-2 ENSEMBLE COMPLETE!")
print(f"Best individual : {max(ensemble_meta['xgb_test_accuracy'], ensemble_meta['lgb_test_accuracy'], ensemble_meta['cat_test_accuracy']):.4f}")
print(f"Ensemble        : {ensemble_acc:.4f}")
print("=" * 55)