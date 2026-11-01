# drybeans_classification
Dry Bean Classification using Machine Learning Models

## A. Problem Statement

The objective of this project is to classify different types of dry beans using machine learning. The dataset contains different measurements related to the size and shape of the beans.

Five classification models are trained using the same dataset. The performance of each model is compared using Accuracy, AUC, Precision, Recall, F1 Score and MCC.

## B. Dataset Description

The Dry Bean Dataset is used for this classification problem. It contains 13,611 samples and 16 input features. All the input features are numerical measurements related to the size and shape of a bean. The target column is `Class`.

There are seven bean classes in the dataset:

- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SEKER
- SIRA

Dataset details:

- Number of samples: 13611
- Number of input features: 16
- Target column: Class
- Number of classes: 7
- Missing values: None
- Duplicate Rows: 68
- Post Removal of Duplicate Rows: 
	- Number of samples: 13543
	- Number of input features: 16 , 
- Problem type: Multiclass classification

## C. Models Used

The following five classification models are used:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (KNN) Classifier
4. Gaussian Naive Bayes Classifier
5. Support Vector Machine (SVM)

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.919158 | 0.993449 | 0.919726 | 0.919158 | 0.919290 | 0.902314 |
| Decision Tree | 0.896641 | 0.936251 | 0.896507 | 0.896641 | 0.896412 | 0.875028 |
| KNN | 0.915467 | 0.981051 | 0.916315 | 0.915467 | 0.915652 | 0.897791 |
| Naive Bayes | 0.763012 | 0.964352 | 0.764673 | 0.763012 | 0.760651 | 0.714296 |
| Support Vector Machine | 0.921004 | 0.993315 | 0.921411 | 0.921004 | 0.921078 | 0.904471 |

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Logistic Regression performed well with an accuracy of 91.92%. It also achieved the highest AUC of 0.993449. Precision, recall and F1 scores are also around 0.92. |
| Decision Tree | Decision Tree achieved 89.66% accuracy. The result is good, but the overall metrics are lower compared to Logistic Regression, KNN and SVM. |
| KNN | KNN achieved 91.55% accuracy. Precision, recall and F1 scores are also around 0.92, so the model performed close to Logistic Regression and SVM. |
| Naive Bayes | Naive Bayes achieved 76.30% accuracy, which is the lowest among the five models. F1 and MCC are also lower compared to the other models. |
| Support Vector Machine | SVM achieved the highest accuracy of 92.10%. It also has the highest precision, recall, F1 and MCC among the five models. |

## Overall, Winner for the Dataset
 **Support Vector Machine (SVM)** gives the best overall result for this dataset. It has the highest accuracy, precision, recall, F1 and MCC. Logistic Regression has a slightly higher AUC, but SVM performs better in most of the evaluation metrics.
 Logistic Regression and KNN also gave good results, with accuracies of 91.92% and 91.55% respectively, and both showed balanced performance across the evaluation metrics. 
