# Module 12 Report Template

## **1. Overview of the Analysis**

### ***Purpose of the analysis.***
This analysis utilises the logistic regression model to separate "Healthy loans" and "High risk loans", with the data provided. As the dataset is imbalanced between the "Healthy loans" and "High risk loans", a random oversampling technique will be applied with the dataset, then this resampled data will be used for the logistic regression model again. Finally, the performances of the model are compared between using original dataset and resampled dataset.

### ***Data and requirement.*** ###
The financial data indicates personal financial condition related to borrowers such as the size of their loans, income, number of accounts, etc. We need to predict whether this loan would be a healthy loan or high risk loan. 

### ***Target variable.*** ###
The variable we are trying to predict is "loan_status". Value of this variable can be either "0" or "1", meaning "healthy loan" or "high eisk loan", respectively. Using "value_counts" function, we can check the number of healthy loan (75,036) and high risk loan (2500) in the data set. We see that there is an imbalance that may affect the performance of the logistic regression model we would employ. 

### ***Machine learning process.*** ###
#### **Using original data** ####
***- Data preparation:*** 
  The dataset will be separated into "features" - "X" and "target" - "y" (loan_status). After this, we will check the number of the value of the "target" to see whether it is imbalanced. Then these data will be splitted for "train" and "test" groups, using the "train_split_test" function. 

***- Logistic regression model implementation:***
  LogisticRegression model is imported from SKLearn library. Then the "train" data will be fitted into the model. Finally, the "test" data will be use with the model to make the prediction. A data frame consisting of the "actual y" (y_test) and "predicted y" will be created to evaluate the performance of the model. 

#### **Using resampled data** ####
The same process will be implemented again as with original data, just that the "train" data used now will be "resampled" by the "RandomOverSampler" function to mitigate the imbalance of the data. Then the performance of the model, using the resampled data, will be evaluated. 

### ***Employed machine learning model and resampling method.*** ###
- LogisticRegression model:
This model is a binary classifier which make prediction (in this case "healthy loan" and "high risk loan") using features from provided data. This regression model is superior than a linear regression since it uses Sigmoid function which is more appropriate for the given problem. 

- Resampling method - RandomOverSampler:
As the data is imbalance between "healthy loan" and "high risk loan", the data distortion affects the performance of the model. In order to mitigate this, we employed RandomOverSampler. It randomly duplicates the data of the minority group "high risk loan", and add them to the "train" data. This balances the dataset and therefore enhance the performance of the model.

## **2. Results**

* Machine Learning Model 1
  * Accuracy score: 95.2% - 95.2% of predictions are correct.  
  * Precision score: 99% of average/total where it scores 100% for "healthy loan" ("0") and 85% for "high-risk loan" ("1"). It measures the proportion of positive identifications was actually correct. The model predicts the "healthy loan" bettern than the "high risk loan".
  * Recall score: 99% of average/total where it scores 99% for "healthy loan" ("0") and 91% for "high-risk loan" ("1"). It measures the proportion of actual positives was identified correctly. The model predicts the "healthy loan" bettern than the "high risk loan". However, the gap between 2 measurements are not significant (8%).

* Machine Learning Model 2:
  * Accuracy score: 99.36% - 99.36% of predictions are correct. 
  * Precision score: 99% of average/total where it scores 100% for "healthy loan" ("0") and 84% for "high-risk loan" ("1"). It measures the proportion of positive identifications was actually correct. The model predicts the "healthy loan" bettern than the "high risk loan".
  * Recall score: 99% of average/total where it scores 99% for "healthy loan" ("0") and 99% for "high-risk loan" ("1"). It measures the proportion of actual positives was identified correctly. The model has the same of accuracy of prediction for both "healthy loan" and "high risk loan".


## **3. Summary**
* Looking at the performance scores, we can see that the logistic regression model using the resampled data has a better performance. It has a remarkable better accuracy score, much lower number of "False Negatives" (which we really care) and the recall scores for both "0" and "1" are the same. 

* It is important to understand which variable we are trying to predict. In this case, it is more important to predict the "high-risk loan" ("1") than the "healthy loan" ("0"). Therefore, the "False Negatives" measurement plays a significant role in the performance evaluation of the model. 

* The balance of the sample proved to be crucial in order to build an effective model. By resampling the data (Random Over Sampling or Random Under Sampling), it can mitigate the imbalance thus improves the performance of the model. 
