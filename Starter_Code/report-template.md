# Module 12 Report Template
 Credit-Risk-Analysis

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
The purpose of the analysing here by using label (loan_status) and features , is to build a predictive model that can accurately classify the 'credi risk' associated with loan applications.Credit risk analysis is a crucial task in the financial industry, particularly for lenders and financial institutions. By leveraging machine learning algorithms, the analysis aims to automate the process of evaluating loan applications and predicting the likelihood of default or non-payment.
Overall, the analysis in machine learning for credit risk aims to improve the accuracy and efficiency of loan evaluations, enhance risk management practices, and support better decision-making in the lending process.

* Explain what financial information the data was on, and what you needed to predict.
We have lebel(Y) as "loan_status" and 7 features (X) include: 'loan_size', 'interest_rate', 'borrower_income', 	'debt_to_income', 'num_of_accounts', 'derogatory_marks', and 	'total_debt'.
As we are analysing leanding data for a financial institution, we want to perdit "Credit Riskk" for lending. The credit risk assessment will help to make better decision for the financial istitution.

* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
The goal is to predict the loan status for each observation in the dataset based on the given features. The features are (alredy identified): 'loan_size', 'interest_rate', 'borrower_income', 	'debt_to_income', 'num_of_accounts', 'derogatory_marks', and 	'total_debt'.

* Describe the stages of the machine learning process you went through as part of this analysis.

The following steps I have followed : 
 Step 1: Created Pandas DataFrame by using Resources "lending_data.csv" 

 Step 2: Create the labels set (`y`)  from the “loan_status” column, and then create the features (`X`) DataFrame from the remaining columns. We have 7 features in the leding dataset. Then seperated 'y and 'X' variables.

 Step 3: Splited the data into training and testing datasets by using 'train_test_split' from 'sklearn.model_selection' in pandas library.

* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

Create a Logistic Regression Model with the Original Data by using following steps:

Step 1: Fit a Logistic Regression model by using the training data (`X_train` and `y_train`) and instituted the Logistic Regression model. 

Step 2: Save the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model.

Step 3: Evaluate the model’s performance by :
 a . Generating  a Confusion matrix.
 b. Generated Classification report.

## ResultsS

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.

Results for the Confusion Matrix :

Confusion Matrix:
[[14926    75]
 [   46   461]]


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.
Results for the Classification Report:

  Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     15001
           1       0.86      0.91      0.88       507

    accuracy                           0.99     15508
   macro avg       0.93      0.95      0.94     15508
weighted avg       0.99      0.99      0.99     15508

## Summary

* . Summarise the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
Summary of the Models' resuts : 

1.Confusing Matrix Model:
Brief explanation of results are as :
True Positives (TP) : This model currectly classified 14926 true positive instances. 
False Positives (FP) : The model wrongly classified 75 instances as positive when they were actually negative.
False Negatives (FN) : The model incorrectly classified 46 instances as negative when they were actually positive.
True Negatives (TN) : The model correctly classified 461 instances as negative.

2.Classification Model :

The class "0" is classified as Healthy Loans  and "1" is classified as High-Risk Loan . 
Breif summary of Healthy Loan are as :
•	Precision: The model has a precision of 1.00 (100%) for the 0 class, which means it correctly identifies all 0 instances out of the total predicted 0 instances.
•	Recall: The model has a recall of 1.00 (100%) for the 0 class, indicating that it correctly identifies all the true 0 instances out of the total actual 0 instances.
•	F1-score: The F1-score, which is a harmonic mean of precision and recall, is also 1.00 (100%)mfor the 0 class.
•	Support: The support value of 15001 indicates the number of actual instances with the 0 label.

Breif summary of High Risk Loans are as :
•	Precision: The precision for the 1 class is 0.86, meaning that out of the predicted 1 instances, 86% are true 1 instances.
•	Recall: The model has a recall of 0.91 for the 1 class, indicating that it correctly identifies 91% of the true 1 instances out of the total actual 1 instances.
•	F1-score: The F1-score for the 1 class is 0.88, which is a balance between precision and recall for this class.
 
* Which one seems to perform best? How do you know it performs best?
Comparison of  the key metrics between the confusion matrix and classification report:
Confusion matrix : 
True Positive (TP): The confusion matrix does not explicitly provide TP values, but we can obtain them from the classification report. From the classification report, the precision and recall values for class 1 are 0.86 and 0.91, respectively. This means there are 461 true positives (TP) for class 1.

False Positive (FP): The confusion matrix shows 75 false positives (FP) for class 1.

False Negative (FN): The confusion matrix shows 46 false negatives (FN) for class 1.

True Negative (TN): The confusion matrix does not explicitly provide TN values, but we can calculate them using the other values. Since the total number of instances for class 1 is 507 and the sum of TP, FP, and FN is 582, we can calculate TN as 15001 - 582 = 14419.

classification report:

Accuracy: The accuracy is reported as 0.99, indicating that the model correctly classified 99% of the instances.

Precision, Recall, and F1-score: The precision and recall values are reported for both class 0 and class 1. The macro-average and weighted-average values for precision, recall, and F1-score are also provided.
Both the confusion matrix and the classification report are useful in evaluating the model's performance and understanding its strengths and weaknesses. The choice of which one to focus on depends on the specific requirements of the problem and the evaluation metrics that are most relevant for the task at hand.
Even though both models has  performed very well, but it is really difficult to judge which is the best one. 
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

Yes, the performance evaluation and the importance of predicting specific classes can vary depending on the problem we are trying to solve. 
We can consider following scenarios:
A : Predicting the 1 class (Positive Class): In certain cases, correctly predicting the positive class (e.g., 1) might be of utmost importance. For example, in a credit risk analysis, correctly identifying potential loan defaulters (positive class) is crucial to minimize financial risks and make informed lending decisions. In such cases, high precision and recall for the positive class would be the primary focus.
B. : Predicting the 0 class (Negative Class): In some situations, accurately predicting the negative class (e.g., 0) may be more important.For instance, in a spam email detection system, correctly classifying legitimate emails as not spam (negative class) is vital to prevent false positives and ensure important messages are not incorrectly flagged. Here, high precision and recall for the negative class would be the main objective.

* If you do not recommend any of the models, please justify your reasoning.

The confusion matrix provides a breakdown of the model's predictions and their alignment with the true labels. It helps evaluate the model's performance in terms of true positives, false positives, false negatives, and true negatives.
The classification report, on the other hand, presents various evaluation metrics such as precision, recall, F1-score, and support for each class. It provides a summary of the model's performance on different metrics.

To determine the best model, we need to compare the performance of different models using these evaluation metrics or other relevant metrics. The choice of the best model would depend on the specific problem, the dataset, the evaluation criteria, and potentially additional factors such as computational complexity or interpretability.

If you have multiple models and their respective confusion matrices or classification reports, we can analyze and compare them to determine which model performs better.