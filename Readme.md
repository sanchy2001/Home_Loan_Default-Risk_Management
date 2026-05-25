# 🏦 Home Loan Default Prediction

## 📌 Project Overview
A machine learning project to predict whether a loan applicant 
will default on their home loan. Built using real-world data 
from Home Credit, this project helps banks make smarter, 
data-driven lending decisions.

## 🎯 Problem Statement
- Predict loan default (1 = Default, 0 = No Default)
- Handle severe class imbalance (91.9% vs 8.1%)
- Identify key factors driving default risk

## 📊 Dataset
- Source: Home Credit Default Risk
- 307,511 loan applicants
- 7 relational data files merged into one master dataset
- 56+ million rows aggregated into 147 features

## 🔧 Tech Stack
- **Language:** Python 3
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, 
  Imbalanced-learn, Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

## 📁 Project Structure

HomeLoanDefault/
│
├── data/                    ← raw and processed CSV files
├── notebooks/
│   └── home_loan_default.ipynb   ← main project notebook
├── README.md
└── .gitignore

## 🔄 Project Workflow

### Phase 1 — Data Understanding
- Loaded 7 relational data files
- Identified 91.9% vs 8.1% class imbalance
- Found 67 columns with missing values

### Phase 2 — Feature Engineering
- Aggregated 56M+ rows from 6 auxiliary files
- Created 25 new features per applicant
- Merged all files into one master dataframe (307,511 × 147)

### Phase 3 — Data Preprocessing
- Dropped columns with >50% missing values (147 → 99 columns)
- Fixed data quality issues (XNA values in CODE_GENDER)
- Imputed missing values (median for numerical, mode for categorical)
- Label encoded 13 categorical columns

### Phase 4 — Modelling
- Applied SMOTE to handle class imbalance
- Trained 3 models and compared using ROC-AUC

## 📈 Model Results

| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | 0.6235 |
| Random Forest | 0.6964 |
| **XGBoost** | **0.7198** ✅ |

**Best Model:** XGBoost with decision threshold = 0.3
- Defaulter Recall improved from 2% → 19% after threshold adjustment
- Nearly 10x improvement in catching real defaulters

## 🔍 Top Default Predictors
1. **FLAG_PHONE** — Applicants without phone numbers are higher risk
2. **WEEKDAY_APPR_PROCESS_START** — Day of application matters
3. **OBS_30_CNT_SOCIAL_CIRCLE** — Social circle payment behaviour
4. **FLAG_DOCUMENT_8 & 6** — Document submission completeness
5. **refused_count** — Previous loan refusal history (engineered feature)

## 💡 Business Recommendations
1. Make phone number mandatory during application
2. Consider social circle risk in loan assessment
3. Enforce strict document submission policies
4. Flag applicants with 3+ previous refusals as high risk
5. Deploy model with 0.3 threshold instead of default 0.5

## 🚀 How to Run

1. Clone the repository

git clone <your-repo-url>

2. Install dependencies

pip install pandas numpy scikit-learn xgboost
imbalanced-learn matplotlib seaborn

3. Download the dataset from Home Credit Default Risk
4. Place CSV files in the `data/` folder
5. Open and run `notebooks/home_loan_default.ipynb`

## 👤 Author
**Sanchit Kumar Mahto**
- B.Tech CSE — SSIPMT Raipur (2023)
- Data Scientist | AI/ML Engineer