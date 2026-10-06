# Credit Card Fraud Detection

Detecting fraudulent card transactions in a dataset where only about 0.17% of transactions are fraud.

## Dataset
Kaggle "Credit Card Fraud Detection" dataset (ULB Machine Learning Group). 284,807 transactions; features V1-V28 are anonymised (PCA), plus Time and Amount. The dataset is not included here because of its size: download it from Kaggle and place `creditcard.csv` next to the notebook.

## What I did
1. Removed 1,081 duplicate rows, leaving 283,726 transactions (473 fraud).
2. Split into train and test sets before scaling or balancing, keeping the fraud ratio the same in both (stratified).
3. Used a plain logistic regression as a baseline. It had 99.9% accuracy but caught only 56 of 95 fraud cases in the test set, so accuracy is misleading here.
4. Compared logistic regression (plain, class-weighted, SMOTE), random forest and gradient boosting using precision, recall, F1 and PR-AUC.
5. Looked at how different probability thresholds change missed fraud vs false alarms.

## Results (test set, threshold 0.5)
| Model | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|
| Random Forest | 0.885 | 0.726 | 0.798 | 0.795 |
| Gradient Boosting | 0.369 | 0.832 | 0.511 | 0.729 |
| LogReg (plain) | 0.848 | 0.589 | 0.696 | 0.692 |
| LogReg + SMOTE | 0.053 | 0.874 | 0.100 | 0.677 |
| LogReg (class weights) | 0.056 | 0.874 | 0.106 | 0.672 |

- Class weights and SMOTE raised recall but caused far too many false alarms (about 1,480 with SMOTE).
- Random forest was best. At a 0.3 threshold it caught 75 of 95 fraud cases with 16 false alarms.
- Most important features: V14, V10, V12, V17, V4.

## Limitations
- Features are anonymised, so I can't explain what drives fraud in business terms.
- The test set has only 95 fraud cases, so results can change with a different split.
- I looked at thresholds on the test set; a stricter version would use a separate validation set.

## How to run
```
pip install pandas numpy matplotlib scikit-learn imbalanced-learn
jupyter notebook fraud_detection.ipynb
```
