import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# 🧭 Load your cleaned dataset
df = pd.read_csv("data/startup_data.csv")

# 🧹 Drop unwanted columns again (precaution if running standalone)
columns_to_drop = [
    'Unnamed: 0', 'Unnamed: 6', 'id', 'object_id', 'state_code.1',
    'name', 'city', 'zip_code', 'labels', 'closed_at'
]
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# 🛠️ Fill missing values if still present
df['age_first_milestone_year'] = df['age_first_milestone_year'].fillna(df['age_first_milestone_year'].median())
df['age_last_milestone_year'] = df['age_last_milestone_year'].fillna(df['age_last_milestone_year'].median())

# 🎯 Encode target: 1 = acquired/IPO (success), 0 = others (fail)
df['status'] = df['status'].apply(lambda x: 1 if x.strip() in ['acquired', 'ipo'] else 0)

# 🚧 Drop date columns for simplicity
df.drop(columns=['founded_at', 'first_funding_at', 'last_funding_at'], inplace=True)

# 📦 Split features and target
X = df.drop('status', axis=1)
y = df['status']

# 🔁 One-hot encode categorical features
X = pd.get_dummies(X)
# Save the columns used during training
joblib.dump(X.columns.tolist(), "models/columns.pkl")


# 🧪 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Align columns in train and test sets
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# ⚙️ Train models
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# 📈 Predict
lr_preds = lr.predict(X_test)
rf_preds = rf.predict(X_test)

# 🧾 Results
print("📊 Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))
print("📊 Random Forest Accuracy:", accuracy_score(y_test, rf_preds))

print("\n📃 Logistic Regression Report:")
print(classification_report(y_test, lr_preds))

print("\n🌲 Random Forest Report:")
print(classification_report(y_test, rf_preds))

# 💾 Save the trained Logistic Regression model
os.makedirs("models", exist_ok=True)
joblib.dump(lr, "models/logistic_model.pkl")
print("✅ Model saved to models/logistic_model.pkl")
