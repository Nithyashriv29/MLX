import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import joblib

# ===============================
# 1. LOAD DATA
# ===============================

train_df = pd.read_csv("../outputs/train_ready.csv", low_memory=False)
test_df = pd.read_csv("../outputs/test_ready.csv", low_memory=False)

# ===============================
# 2. DATE PARSING
# ===============================

for df in [train_df, test_df]:
    df["service_date"] = pd.to_datetime(
        df["service_date"], format="mixed", errors="coerce"
    )
    df.dropna(subset=["service_date"], inplace=True)

    # Date features
    df["service_day"] = df["service_date"].dt.day
    df["service_month"] = df["service_date"].dt.month
    df["service_weekday"] = df["service_date"].dt.weekday

    df.drop(columns=["service_date"], inplace=True)

# ===============================
# 3. TARGET & FEATURES
# ===============================

y = train_df["final_service_units"]
X = train_df.drop(columns=["final_service_units"])

# ===============================
# 4. ENCODE CATEGORICAL FEATURES
# ===============================

cat_cols = [
    "origin_region",
    "destination_region",
    "origin_hub_tier",
    "destination_hub_tier"
]

for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

# ===============================
# 5. TRAIN–VALIDATION SPLIT (OPTIONAL)
# ===============================

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 6. LIGHTGBM REGRESSOR
# ===============================

model = lgb.LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42
)

# ===============================
# 7. TRAIN MODEL (NO EARLY STOPPING)
# ===============================

model.fit(X_train, y_train)

# ===============================
# 8. VALIDATION
# ===============================

val_preds = model.predict(X_val)
mae = mean_absolute_error(y_val, val_preds)
print("LightGBM Validation MAE:", mae)

# ===============================
# 9. TRAIN FINAL MODEL ON FULL DATA
# ===============================

model.fit(X, y)
joblib.dump(model, "../outputs/lightgbm_demand_model.pkl")

# ===============================
# 10. PREDICT ON TEST DATA
# ===============================

service_keys = test_df["service_key"]
X_test = test_df.drop(columns=["service_key"])

test_preds = model.predict(X_test)

# 🔴 ROUND OFF FINAL OUTPUT
test_preds = np.round(test_preds).astype(int)

# ===============================
# 11. CREATE SUBMISSION FILE
# ===============================

submission = pd.DataFrame({
    "service_key": service_keys,
    "final_service_units": test_preds
})

submission.to_csv("../outputs/submission.csv", index=False)

print("submission.csv created successfully (rounded integers).")
print(submission.head())