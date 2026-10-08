# 💳 Executive Credit Risk & Loan Default Assessment

An end-to-end Machine Learning project that leverages an interpretable **Decision Tree Classifier** to evaluate bank loan applicant risk profiles. The project includes data preprocessing pipelines, model training notebooks, and an aesthetic, high-performance **Streamlit** web application featuring a deep indigo-violet glassmorphism UI.

---

## 🚀 Project Overview

Financial institutions must carefully assess credit risk before approving loans to minimize default rates while maintaining transparent, explainable decision-making. This project predicts whether an applicant is likely to default on a loan based on key financial attributes.

* **Model Type:** Decision Tree Classifier (Supervised Classification)
* **Frontend UI:** Streamlit (Custom Dark Indigo/Violet Theme, Metric Cards, Responsive Layout)
* **Dataset Source:** Kaggle Credit Risk / Loan Default Dataset

---

## 🛠️ Tech Stack & Libraries

* **Python 3.8+**
* **Pandas & NumPy:** Data manipulation, feature engineering, and missing value imputation.
* **Scikit-Learn:** Model training, hyperparameter tuning, evaluation metrics (`accuracy_score`, `classification_report`), and Decision Tree implementation.
* **Joblib:** Model persistence (saving and loading trained models and features).
* **Streamlit:** Interactive web interface and deployment framework.

📊 Features & Workflow
Data Preprocessing: Handles missing numerical values using median imputation and encodes categorical features.

Feature Engineering: Automatically computes vital indicators such as the Debt-to-Income Ratio (requested_loan_amount / annual_income).

Interpretable ML: Utilizes a constrained Decision Tree (max_depth=5) to ensure rules remain transparent and understandable for financial auditors.

Executive UI: Built with custom CSS featuring smooth gradient backgrounds, glassmorphism metric cards, hidden default Streamlit headers, and responsive columns.

📈 Model Evaluation Metrics
The model is evaluated using standard classification metrics:

Accuracy Score

Precision & Recall (to minimize false approvals of high-risk loans)

Classification Report
