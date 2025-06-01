# Loan Default Prediction

This project predicts the likelihood of loan defaults using various machine learning models, including Logistic Regression, Random Forest, and XGBoost. After thorough evaluation, the Random Forest Classifier was selected as the final model due to its superior performance.

## Dataset
The dataset is sourced from Kaggle:
[Default of Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

### Key Features:
- **LIMIT_BAL**: Credit limit for each customer.
- **SEX**: Gender (1 = male, 2 = female).
- **EDUCATION**: Education level.
- **MARRIAGE**: Marital status.
- **PAY_0 to PAY_6**: History of past payment statuses.
- **BILL_AMT1 to BILL_AMT6**: Bill statement amounts for the past six months.
- **PAY_AMT1 to PAY_AMT6**: Payment amounts for the past six months.

### Target:
- **default.payment.next.month**: Indicates whether a customer defaulted (1 = yes, 0 = no).

## Models Evaluated
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### Results
After evaluating multiple models, the Random Forest Classifier achieved the best performance:

#### Random Forest Performance:
- **Accuracy**: 79%
- **ROC-AUC Score**: 0.77
- **Classification Report**:

```
               precision    recall  f1-score   support

           0       0.87      0.86      0.86      4673
           1       0.52      0.56      0.54      1327

    accuracy                           0.79      6000
   macro avg       0.70      0.71      0.70      6000
weighted avg       0.80      0.79      0.79      6000
```

#### ROC Curve
The ROC Curve shows the model's trade-off between sensitivity and specificity.

---

## How to Run
1. Clone this repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the script `loan_default_prediction.py`.

---

## License
This project is open-source and available under the MIT License.
