# Credit Card Approval Classification Using Machine Learning Classification Models

## Problem Statement

The objective of this project is to build and compare multiple machine learning classification models for Credit Card Approval Classification. The models are trained on a credit card fillup dataset and evaluated using various performance metrics such as Accuracy, AUC Score, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

A Streamlit web application is developed to allow users to upload test data, select a machine learning model, and view model evaluation results interactively.

---

## Dataset Description

### Dataset Name
Credit Card Filing details Dataset

### Source
Kaggle

### Dataset Characteristics

- Total Records: 1549
- Total Features: 20
- Target Variable: label
- Class 0 = Application Approved
- Class 1 = Application Rejected

### Features

The dataset contains:
```text
-Ind_ID: Client ID
-Gender: Gender information
-Car_owner: Having car or not
-Propert_owner: Having property or not
-Children: Count of children
-Annual_income: Annual income
-Type_Income: Income type
-Education: Education level
-Marital_status: Marital_status
-Housing_type: Living style
-Birthday_count: Use backward count from current day (0), -1 means yesterday.
-Employed_days: Start date of employment. Use backward count from current day (0). Positive value means, individual is currently unemployed.
-Mobile_phone: Any mobile phone
-Work_phone: Any work phone
-Phone: Any phone number
-EMAIL_ID: Any email ID
-Type_Occupation: Occupation
-Family_Members: Family size
-Another data set (Credit_card_label.csv) contains two key pieces of information
-ID: The joining key between application data and credit status data, same is Ind_ID
-Label: 0 is application approved and 1 is application rejected
```

The dataset satisfies the assignment requirements:

- Number of Instances > 500
- Number of Features > 12

---

## Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── data_preprocessing.py
│   ├── create_train_test.py
│   ├── train_data.csv
│   └── test_data.csv
│
├── models/
│   ├── train_all_models.py
│   ├── logistic_regression_model.pkl
│   ├── decision_tree_model.pkl
│   ├── knn_model.pkl
│   ├── naive_bayes_model.pkl
│   ├── random_forest_model.pkl
│   └── encoders.pkl
│
└── utils/
    └── metrics.py
```

## Github Repository Link Repository URL:
URL :

## Streamlit Application Link
URL :


## Machine Learning Models Implemented
Logistic Regression
Decision Tree Classifier
K-Nearest Neighbors (KNN)
Gaussian Naive Bayes
Random Forest Classifier
Evaluation Metrics

## The following evaluation metrics are used:
Accuracy
AUC Score
Precision
Recall
F1 Score
Matthews Correlation Coefficient (MCC)



| Model               |   Accuracy |   AUC Score |   Precision |   Recall |   F1 Score |   MCC Score |
|:--------------------|-----------:|------------:|------------:|---------:|-----------:|------------:|
| Logistic Regression |   0.887097 |    0.595117 |    0        | 0        |   0        |    0        |
| Decision Tree       |   0.851613 |    0.691948 |    0.377778 | 0.485714 |   0.425    |    0.344893 |
| KNN                 |   0.880645 |    0.702234 |    0.4      | 0.114286 |   0.177778 |    0.165626 |
| Naive Bayes         |   0.887097 |    0.500779 |    0        | 0        |   0        |    0        |
| Random Forest       |   0.935484 |    0.814182 |    0.941176 | 0.457143 |   0.615385 |    0.630414 |

## Observation on result

| Model                         | Observation                                                                                                                                                                                                                                                                                                      |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**       | Achieved **88.71% accuracy**, but failed to identify any positive-class instances, resulting in **0 precision, recall, F1-score, and MCC**. Although the accuracy is high, the model is ineffective for minority class prediction and is likely biased toward the majority class.                                |
| **Decision Tree**             | Produced **85.16% accuracy** with a balanced ability to detect positive cases. It achieved **48.57% recall** and **42.50% F1-score**, indicating moderate classification performance. The **AUC score of 0.692** suggests reasonable discriminative capability.                                                  |
| **K-Nearest Neighbors (KNN)** | Obtained **88.06% accuracy** and an **AUC of 0.702**, but suffered from very low recall (**11.43%**) and F1-score (**17.78%**). The model identifies very few positive cases despite acceptable overall accuracy.                                                                                                |
| **Naïve Bayes**               | Similar to Logistic Regression, achieved **88.71% accuracy** but failed to classify any positive instances, leading to **zero precision, recall, F1-score, and MCC**. The **AUC score of 0.501** indicates near-random classification capability.                                                                |
| **Random Forest**             | Delivered the **best overall performance** with **93.55% accuracy**, **0.814 AUC**, **94.12% precision**, and the highest **F1-score (61.54%)** and **MCC (0.630)**. It effectively balances prediction accuracy and positive-class detection, making it the most reliable model among all evaluated approaches. |


## Overall Conclusion

Among the five evaluated models, Random Forest demonstrated the strongest performance across nearly all evaluation metrics, achieving the highest accuracy (93.55%), AUC score (0.814), F1-score (61.54%), and MCC score (0.630). This indicates superior predictive power and balanced classification capability.

Although Logistic Regression and Naïve Bayes reported relatively high accuracy (88.71%), they completely failed to identify positive-class instances, making them unsuitable for practical deployment in this classification task. Decision Tree showed moderate performance with a reasonable balance between precision and recall, while KNN achieved acceptable accuracy but struggled to detect positive cases effectively.

Therefore, Random Forest is the recommended model for this dataset, as it provides the best trade-off between overall accuracy, class discrimination, and reliable positive-class prediction.



```python

```




```python

```
