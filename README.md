# Predictive Churn Analysis

## Submission contents
- predictive_churn_analysis_report.pdf — main report
- predictive_churn_analysis.ipynb — reproducible Jupyter Notebook
- predictive_churn_model.py — reproducible Python script
- customer_churn_risk_scores.csv — exported customer churn probabilities
- customer_churn_sample(1).csv — source dataset

## Pipeline
1. Binary churn target encoding
2. 80/20 stratified train-test split
3. One-Hot Encoding for categorical fields
4. MinMax scaling for numeric fields
5. Logistic Regression with class balancing
6. Precision, Recall, F1 and ROC-AUC evaluation
7. Final-model customer churn probability export

## Important limitation
The uploaded dataset contains only 15 records. Test-set metrics are therefore directional and not production-ready.
