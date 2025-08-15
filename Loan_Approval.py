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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

# Step 0: Project Flowchart
plt.figure(figsize=(8, 10))

# Title with extra gap
plt.title("Figure 3.1 – Project Flowchart", fontsize=14, weight='bold', pad=40)  # pad adds gap

flow_steps = [
    "Data Collection",
    "Data Preprocessing",
    "EDA (Exploratory Data Analysis)",
    "Feature Selection",
    "Model Training\n(Logistic Regression & Decision Tree)",
    "Model Evaluation",
    "Model Comparison",
    "Prediction",
    "Deployment (Future)"
]

for i, step in enumerate(flow_steps):
    plt.text(0.5, 1 - (i * 0.1), step,
             ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="skyblue", edgecolor="black"))
    if i < len(flow_steps) - 1:
        plt.arrow(0.5, 1 - (i * 0.1) - 0.05, 0, -0.05, head_width=0.02, head_length=0.02, fc='black', ec='black')

plt.axis('off')
plt.show()

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

# Step 3: EDA
print("\n--- Class Distribution ---")
print(df_clean['LoanApproved'].value_counts())
sns.countplot(x='LoanApproved', data=df_clean)
plt.title('Loan Approval Distribution')
plt.xlabel('Loan Approved (1 = Yes, 0 = No)')
plt.ylabel('Number of Applicants')
plt.show()

print("\n--- Correlation Matrix ---")
print(df_clean.corr())
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

df_clean[num_cols].hist(bins=20, figsize=(8, 6))
plt.suptitle('Distribution of Numeric Features')
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

# Step 7: ROC Curve Comparison
y_prob_log = log_model.predict_proba(X_test)[:, 1]
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log, pos_label=1)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt, pos_label=1)

plt.figure(figsize=(8, 6))
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC={auc(fpr_log, tpr_log):.2f})')
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC={auc(fpr_dt, tpr_dt):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.show()


# Step 8: Final Comparison
print("\n--- Final Model Comparison ---")
print(f"Logistic Regression Accuracy: {log_acc:.2f}")
print(f"Decision Tree Accuracy: {dt_acc:.2f}")
if log_acc > dt_acc:
    print("Conclusion: Logistic Regression performed better.\nExplanation: Logistic Regression may generalize better on small or linearly separable data.")
elif dt_acc > log_acc:
    print("Conclusion: Decision Tree performed better.\nExplanation: The model likely captured complex feature interactions that Logistic Regression could not.")
else:
    print("Conclusion: Both models performed equally well.\nExplanation: Try cross-validation or advanced models (e.g., Random Forest) for clearer insights.")

