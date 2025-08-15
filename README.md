# 📌 Loan Approval Predictor – Machine Learning Project

## 📖 Overview
The **Loan Approval Predictor** is a machine learning project designed to predict whether a loan application will be **approved** or **rejected** based on applicant details.  

This project follows a **complete end-to-end ML pipeline**:
1. Data Collection & Cleaning  
2. Exploratory Data Analysis (EDA)  
3. Model Training – Logistic Regression & Decision Tree  
4. Model Evaluation & Comparison  
5. Visualization (Heatmap, ROC Curves, Histograms)  

---

## ✨ Features
- **Data Preprocessing**: Missing value handling, label encoding, numeric scaling.  
- **EDA**: Class distribution, correlation analysis, numeric feature visualization.  
- **Machine Learning Models**: Logistic Regression & Decision Tree Classifier.  
- **Performance Metrics**: Accuracy, Precision, Recall, F1-score.  
- **Visualization**: Flowchart, ROC curves, confusion matrices, histograms.  
- **Model Comparison**: Side-by-side performance evaluation.  

---

## 📂 Dataset
The dataset `loan_approval_.xlsx` contains:
| Feature           | Description |
|-------------------|-------------|
| ApplicantIncome   | Monthly income of the applicant |
| LoanAmount        | Requested loan amount |
| CreditScore       | Creditworthiness score |
| Education         | Applicant's education level |
| SelfEmployed      | Employment type |
| LoanApproved      | Target variable (1 = Approved, 0 = Rejected) |

---

## 🛠 Workflow
**Figure 3.1 – Project Flowchart**
```bash
Data Collection
      ↓
Data Preprocessing
      ↓
EDA (Exploratory Data Analysis)
      ↓
Feature Selection
      ↓
Model Training (Logistic Regression & Decision Tree)
      ↓
Model Evaluation
      ↓
Model Comparison
      ↓
Prediction
      ↓
Deployment (Future Scope)
```
---

## 🚀 How to Run

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/Loan-Approval-Predictor.git
cd Loan-Approval-Predictor
```

2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
3️⃣ Add the Dataset

* Place loan_approval_.xlsx in the project directory.

4️⃣ Run the Script
```bash
python loan_predictor.py
```

## 📊 Model Evaluation
* Model	Accuracy	Precision	Recall	F1-score
* Logistic Regression	~50%	0.52	0.50	0.48
* Decision Tree	~44%	0.45	0.44	0.43

✅ Logistic Regression performed slightly better in accuracy.

✅ Decision Tree provided better interpretability for feature-based decisions.

## 📈 Visualizations

* 📊 Loan Approval Distribution – Bar chart for approvals vs. rejections.

* 🔥 Correlation Heatmap – Feature relationships.

* 📉 Numeric Feature Distributions – Histograms for income, loan amount, credit score.

* 🎯 ROC Curve Comparison – Logistic Regression vs. Decision Tree.

## 🖥 Technologies Used

* Python 3.x

* Pandas – Data manipulation

* NumPy – Numerical operations

* Matplotlib & Seaborn – Data visualization

* scikit-learn – Machine learning algorithms

## 🔮 Future Scope

* Implement Random Forest, XGBoost, LightGBM for better accuracy.

* Perform hyperparameter tuning for optimal results.

* Add feature engineering for improved predictions.

* Expand dataset with real-world banking data.

* Deploy as a Flask/Django web app for real-time predictions.

* Ensure fairness, transparency & compliance in decision-making.

## 📜 License

* This project is licensed under the MIT License.

## ✍🏼 Author

 **Utkarsh Pandey**
 
* Category: Machine Learning | FinTech | Binary Classification
