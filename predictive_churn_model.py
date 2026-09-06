import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("customer_churn_sample(1).csv")
df["ChurnTarget"] = (df["Churn"] == "Yes").astype(int)

X = df.drop(columns=["Churn", "ChurnTarget", "CustomerID"])
y = df["ChurnTarget"]

categorical = X.select_dtypes(include=["object"]).columns.tolist()
numeric = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", MinMaxScaler())
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
])
model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print("Precision:", precision_score(y_test, pred, zero_division=0))
print("Recall:", recall_score(y_test, pred, zero_division=0))
print("F1:", f1_score(y_test, pred, zero_division=0))
if len(y_test.unique()) == 2:
    print("ROC-AUC:", roc_auc_score(y_test, prob))

# Final model for customer risk scores
model.fit(X, y)
df["ChurnProbability"] = model.predict_proba(X)[:, 1]
df["RiskLevel"] = pd.cut(
    df["ChurnProbability"], bins=[-0.001,0.33,0.66,1.001],
    labels=["Low","Medium","High"]
)
df[["CustomerID","Churn","ChurnProbability","RiskLevel"]].to_csv(
    "customer_churn_risk_scores.csv", index=False
)
