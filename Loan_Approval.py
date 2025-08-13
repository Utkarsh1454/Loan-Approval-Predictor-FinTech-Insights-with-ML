# Loan Approval Predictor - Complete Project with Final Report

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load Data
df = pd.read_excel("loan_approval_.xlsx")

# Step 2: Data Cleaning
df_clean = df.copy()
num_cols = ['ApplicantIncome', 'LoanAmount', 'CreditScore']
for col in num_cols:
    df_clean[col].fillna(df_clean[col].median(), inplace=True)
df_clean.dropna(subset=['LoanApproved'], inplace=True)
df_clean['LoanApproved'] = df_clean['LoanApproved'].astype(int)

# Encode categorical features
le_edu = LabelEncoder()
le_self_emp = LabelEncoder()
df_clean['Education'] = le_edu.fit_transform(df_clean['Education'])
df_clean['SelfEmployed'] = le_self_emp.fit_transform(df_clean['SelfEmployed'])

# Step 3: Exploratory Data Analysis
print("\n--- Class Distribution ---")
print(df_clean['LoanApproved'].value_counts())

print("\n--- Correlation Matrix ---")
print(df_clean.corr())

sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Step 4: Feature and Target Selection
X = df_clean[['ApplicantIncome', 'LoanAmount', 'CreditScore', 'Education', 'SelfEmployed']]
y = df_clean['LoanApproved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
log_acc = accuracy_score(y_test, log_preds)
log_report = classification_report(y_test, log_preds)

print("\n--- Logistic Regression Report ---")
print("Accuracy:", log_acc)
print(log_report)

# Step 6: Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_preds = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)
dt_report = classification_report(y_test, dt_preds)

print("\n--- Decision Tree Report ---")
print("Accuracy:", dt_acc)
print(dt_report)

# Step 7: Final Comparison Report
print("\n--- Final Model Comparison ---")
print(f"Logistic Regression Accuracy: {log_acc:.2f}")
print(f"Decision Tree Accuracy: {dt_acc:.2f}")

if log_acc > dt_acc:
    print("Conclusion: Logistic Regression performed better.\n")
    print("Explanation: Logistic Regression may generalize better on small or linearly separable data.")
elif dt_acc > log_acc:
    print("Conclusion: Decision Tree performed better.\n")
    print("Explanation: The model likely captured complex feature interactions that Logistic Regression could not.")
else:
    print("Conclusion: Both models performed equally well.\n")
    print("Explanation: Try cross-validation or advanced models (e.g., Random Forest) for clearer insights.")
